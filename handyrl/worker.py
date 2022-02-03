# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# worker and gather

import random
import threading
import time
import functools
from socket import gethostname
from collections import deque
import multiprocessing as mp
import pickle
import copy

from .environment import prepare_env, make_env
from .connection import QueueCommunicator
from .connection import send_recv, open_multiprocessing_connections
from .connection import connect_socket_connection, accept_socket_connections
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper, RandomModel


class Worker:
    def __init__(self, args, conn, wid):
        print('opened worker %d' % wid)
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = -1, None

        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)

    def __del__(self):
        print('closed worker %d' % self.worker_id)

    def _gather_models(self, model_ids):
        model_pool = {}
        for model_id in model_ids:
            if model_id not in model_pool:
                if model_id < 0:
                    model_pool[model_id] = None
                elif model_id == self.latest_model[0]:
                    # use latest model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # get model from server
                    model = pickle.loads(send_recv(self.conn, ('model', model_id)))
                    if model_id == 0:
                        # use random model
                        self.env.reset()
                        obs = self.env.observation(self.env.players()[0])
                        model = RandomModel(model, obs)
                    model_pool[model_id] = ModelWrapper(model)
                    # update latest model
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def run(self):
        while True:
            args = send_recv(self.conn, ('args', None))
            role = args['role']

            models = {}
            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids)

                # make dict of models
                for p, model_id in args['model_id'].items():
                    models[p] = model_pool[model_id]

            if role == 'g':
                episode = self.generator.execute(models, args)
                send_recv(self.conn, ('episode', episode))
            elif role == 'e':
                result = self.evaluator.execute(models, args)
                send_recv(self.conn, ('result', result))


def make_worker_args(args, base_wid, wid, conn):
    return args, conn, base_wid + wid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gather_id, base_worker_id, num_workers):
        print('started gather %d' % gather_id)
        super().__init__()
        self.gather_id = gather_id
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        worker_conns = open_multiprocessing_connections(
            num_workers,
            open_worker,
            functools.partial(make_worker_args, args, base_worker_id)
        )

        for conn in worker_conns:
            self.add_connection(conn)

        self.args_buf_len = 1 + len(worker_conns) // 4
        self.result_buf_len = 1 + len(worker_conns) // 4

    def __del__(self):
        print('finished gather %d' % self.gather_id)

    def run(self):
        while True:
            conn, (command, args) = self.recv()
            if command == 'args':
                # When requested arguments, return buffered outputs
                if len(self.args_queue) == 0:
                    # get multiple arguments from server and store them
                    self.server_conn.send((command, [None] * self.args_buf_len))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                # answer data request as soon as possible
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                # return flag first and store data
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.result_buf_len:
                    # send datum to server after buffering certain number of datum
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map = {}
                    self.result_send_cnt = 0


def gather_loop(args, conn, gather_id):
    n_pro, n_ga = args['worker']['num_parallel'], args['worker']['num_gathers']
    n_pro_w = (n_pro // n_ga) + int(gather_id < n_pro % n_ga)
    args['worker']['num_parallel_per_gather'] = n_pro_w
    base_worker_id = 0

    if conn is None:
        # entry
        conn = connect_websocket_connection(args['worker']['tunnel_address'], 8081)
        conn.send(('entry', args['worker']))
        args = conn.recv()

        if gather_id == 0:  # call once at every machine
             print(args)
             prepare_env(args['env'])
        base_worker_id = args['worker'].get('base_worker_id', 0)

    try:
        gather = Gather(args, conn, gather_id, base_worker_id, n_pro_w)
        gather.run()
    finally:
        gather.shutdown()


class WorkerCluster(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        # open local connections
        if 'num_gathers' not in self.args['worker']:
            self.args['worker']['num_gathers'] = 1 + max(0, self.args['worker']['num_parallel'] - 1) // 16
        for i in range(self.args['worker']['num_gathers']):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
            conn1.close()
            self.add_connection(conn0)


class WorkerServer(QueueCommunicator):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.total_worker_count = 0

    def run(self):
        # prepare listening connections
        def worker_server(port):
            print('started worker server %d' % port)
            conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
            while not self.shutdown_flag:
                conn = next(conn_acceptor)
                if conn is not None:
                    self.add_connection(conn)
            print('finished worker server')

        # use thread list of super class
        self.threads.append(threading.Thread(target=worker_server, args=(9998,)))
        self.threads[-1].start()


class RemoteWorkerCluster:
    def __init__(self, args):
        args['address'] = gethostname()
        if 'num_gathers' not in args:
            args['num_gathers'] = 1 + max(0, args['num_parallel'] - 1) // 16

        self.args = args

    def run(self):
        # open worker
        process = []
        try:
            for i in range(self.args['num_gathers']):
                p = mp.Process(target=gather_loop, args=({'worker': self.args}, None, i))
                p.start()
                process.append(p)
            while True:
                time.sleep(100)
        finally:
            for p in process:
                p.terminate()


import base64
import queue
import socket
from websocket import create_connection
from websocket_server import WebsocketServer


class WebsocketConnection:
    def __init__(self, conn):
        self.conn = conn

    @staticmethod
    def dumps(data):
        return base64.b85encode(pickle.dumps(data))

    @staticmethod
    def loads(message):
        return pickle.loads(base64.b85decode(message))

    def send(self, data):
        message = self.dumps(data)
        self.conn.send(message)

    def recv(self):
        message = self.conn.recv()
        return self.loads(message)

    def close(self):
        self.conn.close()


def connect_websocket_connection(host, port):
    host = socket.gethostbyname(host)
    conn = create_connection('ws://%s:%d' % (host, port))
    return WebsocketConnection(conn)


class TunnelServer(WebsocketServer):
    def __init__(self, args):
        super().__init__(port=8081, host='0.0.0.0')
        self.args = args
        self.lock = threading.Lock()

    def run(self):
        self.conn = connect_socket_connection(self.args['worker']['server_address'], 9998)

        self.set_fn_new_client(self._new_client)
        self.set_fn_message_received(self._message_received)
        self.run_forever()

    @staticmethod
    def _new_client(client, server):
        print('New client {}:{} has joined.'.format(client['address'][0], client['address'][1]))

    @staticmethod
    def _message_received(client, server, message):
        data = WebsocketConnection.loads(message)
        server.lock.acquire()
        reply_data= send_recv(server.conn, data)
        server.lock.release()
        reply_message = WebsocketConnection.dumps(reply_data)
        server.send_message(client, reply_message)


def worker_main(args):
    # offline generation worker
    worker = RemoteWorkerCluster(args=args['worker_args'])
    worker.run()

def worker_tunnel_main(args):
    # offline generation worker
    worker = TunnelServer(args={'worker': args['worker_args']})
    worker.run()
