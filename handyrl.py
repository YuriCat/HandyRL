# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import numpy as np


def map_r(x, callback_fn=None):
    # recursive map function
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn)) for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


def bimap_r(x, y, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(bimap_r(xx, y[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, bimap_r(xx, y[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y) if callback_fn is not None else None


def trimap_r(x, y, z, callback_fn=None):
    if isinstance(x, (list, tuple)):
        return type(x)(trimap_r(xx, y[i], z[i], callback_fn) for i, xx in enumerate(x))
    elif isinstance(x, dict):
        return type(x)((key, trimap_r(xx, y[key], z[key], callback_fn)) for key, xx in x.items())
    return callback_fn(x, y, z) if callback_fn is not None else None


def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1)
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1))
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1)
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1))
                for key2 in x_front
            )
    return x


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


## connection

import io
import time
import struct
import socket
import pickle
import threading
import queue
import multiprocessing as mp
import multiprocessing.connection as connection


def send_recv(conn, sdata):
    conn.send(sdata)
    rdata = conn.recv()
    return rdata


class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fileno(self):
        return self.conn.fileno()

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            if len(chunk) == 0:
                raise ConnectionResetError
            size -= len(chunk)
            buf.write(chunk)
        return buf

    def recv(self):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf):
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg):
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack("!i", n)
        if n > 16384:
            chunks = [header, buf]
        elif n > 0:
            chunks = [header + buf]
        else:
            chunks = [header]
        for chunk in chunks:
            self._send(chunk)


def open_socket_connection(port, reuse=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR,
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
    )
    sock.bind(('', int(port)))
    return sock


def accept_socket_connection(sock):
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


def listen_socket_connections(n, port):
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def connect_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, int(port)))
    except ConnectionRefusedError:
        print('failed to connect %s %d' % (host, port))
    return PickledConnection(sock)


def accept_socket_connections(port, timeout=None, maxsize=1024):
    sock = open_socket_connection(port)
    sock.listen(maxsize)
    sock.settimeout(timeout)
    cnt = 0
    while cnt < maxsize:
        conn = accept_socket_connection(sock)
        if conn is not None:
            cnt += 1
        yield conn


def open_multiprocessing_connections(num_process, target, args_func):
    # open connections
    s_conns, g_conns = [], []
    for _ in range(num_process):
        conn0, conn1 = mp.Pipe(duplex=True)
        s_conns.append(conn0)
        g_conns.append(conn1)

    # open workers
    for i, conn in enumerate(g_conns):
        mp.Process(target=target, args=args_func(i, conn)).start()
        conn.close()

    return s_conns


class MultiProcessJobExecutor:
    def __init__(self, func, send_generator, num_workers, postprocess=None, num_receivers=1):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.num_receivers = num_receivers
        self.conns = []
        self.waiting_conns = queue.Queue()
        self.shutdown_flag = False
        self.output_queue = queue.Queue(maxsize=8)
        self.threads = []

        for i in range(num_workers):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=func, args=(conn1, i)).start()
            conn1.close()
            self.conns.append(conn0)
            self.waiting_conns.put(conn0)

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.output_queue.get()

    def start(self):
        self.threads.append(threading.Thread(target=self._sender))
        for i in range(self.num_receivers):
            self.threads.append(threading.Thread(target=self._receiver, args=(i,)))
        for thread in self.threads:
            thread.start()

    def _sender(self):
        print('start sender')
        while not self.shutdown_flag:
            data = next(self.send_generator)
            while not self.shutdown_flag:
                try:
                    conn = self.waiting_conns.get(timeout=0.3)
                    conn.send(data)
                    break
                except queue.Empty:
                    pass
        print('finished sender')

    def _receiver(self, index):
        print('start receiver %d' % index)
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = connection.wait(conns)
            for conn in tmp_conns:
                data = conn.recv()
                self.waiting_conns.put(conn)
                if self.postprocess is not None:
                    data = self.postprocess(data)
                while not self.shutdown_flag:
                    try:
                        self.output_queue.put(data, timeout=0.3)
                        break
                    except queue.Full:
                        pass
        print('finished receiver %d' % index)


class QueueCommunicator:
    def __init__(self, conns=[]):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = {}
        self.conn_index = 0
        for conn in conns:
            self.add_connection(conn)
        self.shutdown_flag = False
        self.threads = [
            threading.Thread(target=self._send_thread),
            threading.Thread(target=self._recv_thread),
        ]
        for thread in self.threads:
            thread.start()

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.input_queue.get()

    def send(self, conn, send_data):
        self.output_queue.put((conn, send_data))

    def add_connection(self, conn):
        self.conns[conn] = self.conn_index
        self.conn_index += 1

    def disconnect(self, conn):
        print('disconnected')
        self.conns.pop(conn, None)

    def _send_thread(self):
        while not self.shutdown_flag:
            try:
                conn, send_data = self.output_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                conn.send(send_data)
            except ConnectionResetError:
                self.disconnect(conn)
            except BrokenPipeError:
                self.disconnect(conn)

    def _recv_thread(self):
        while not self.shutdown_flag:
            conns = connection.wait(self.conns, timeout=0.3)
            for conn in conns:
                try:
                    recv_data = conn.recv()
                except ConnectionResetError:
                    self.disconnect(conn)
                    continue
                except EOFError:
                    self.disconnect(conn)
                    continue
                while not self.shutdown_flag:
                    try:
                        self.input_queue.put((conn, recv_data), timeout=0.3)
                        break
                    except queue.Full:
                        pass


## environment

import importlib


ENVS = {
    'TicTacToe':         'envs.tictactoe',
    'Geister':           'envs.geister',
    'ParallelTicTacToe': 'envs.parallel_tictactoe',
    'HungryGeese':       'envs.kaggle.hungry_geese',
}


def prepare_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    elif hasattr(env_module, 'prepare'):
        env_module.prepare()


def make_env(env_args):
    env_name = env_args['env']
    env_source = ENVS.get(env_name, env_name)
    env_module = importlib.import_module(env_source)

    if env_module is None:
        print("No environment %s" % env_name)
    else:
        return env_module.Environment(env_args)


class BaseEnvironment:
    def __init__(self, args={}):
        pass

    def __str__(self):
        return ''

    #
    # Should be defined in all games
    #
    def reset(self, args={}):
        raise NotImplementedError()

    #
    # Should be defined in all games except you implement original step() function
    #
    def play(self, action, player):
        raise NotImplementedError()

    #
    # Should be defined in games which has simultaneous trainsition
    #
    def step(self, actions):
        for p, action in actions.items():
            if action is not None:
                self.play(action, p)

    #
    # Should be defined if you use multiplayer sequential action game
    #
    def turn(self):
        return 0

    #
    # Should be defined if you use multiplayer simultaneous action game
    #
    def turns(self):
        return [self.turn()]

    #
    # Should be defined in all games
    #
    def terminal(self):
        raise NotImplementedError()

    #
    # Should be defined if you use immediate reward
    #
    def reward(self):
        return {}

    #
    # Should be defined in all games
    #
    def outcome(self):
        raise NotImplementedError()

    #
    # Should be defined in all games
    #
    def legal_actions(self, player):
        raise NotImplementedError()

    #
    # Should be defined if you use multiplayer game or add name to each player
    #
    def players(self):
        return [0]

    #
    # Should be defined in all games
    #
    def observation(self, player=None):
        raise NotImplementedError()

    #
    # Should be defined if you encode action as special string
    #
    def action2str(self, a, player=None):
        return str(a)

    #
    # Should be defined if you encode action as special string
    #
    def str2action(self, s, player=None):
        return int(s)

    #
    # Should be defined if you use network battle mode
    #
    def diff_info(self, player=None):
        return ''

    #
    # Should be defined if you use network battle mode
    #
    def update(self, info, reset):
        raise NotImplementedError()


## model

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F


def to_torch(x):
    return map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous() if x is not None else None)


def to_numpy(x):
    return map_r(x, lambda x: x.detach().numpy() if x is not None else None)


def to_gpu(data):
    return map_r(data, lambda x: x.cuda() if x is not None else None)


# model wrapper class

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, 'init_hidden'):
            return self.model.init_hidden(batch_size)
        return None

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def inference(self, x, hidden, **kwargs):
        # numpy array -> numpy array
        if hasattr(self.model, 'inference'):
            return self.model.inference(x, hidden, **kwargs)

        self.eval()
        with torch.no_grad():
            xt = map_r(x, lambda x: torch.from_numpy(np.array(x)).contiguous().unsqueeze(0) if x is not None else None)
            ht = map_r(hidden, lambda h: torch.from_numpy(np.array(h)).contiguous().unsqueeze(0) if h is not None else None)
            outputs = self.forward(xt, ht, **kwargs)
        return map_r(outputs, lambda o: o.detach().numpy().squeeze(0) if o is not None else None)


# simple model

class RandomModel(nn.Module):
    def __init__(self, model, x):
        super().__init__()
        wrapped_model = ModelWrapper(model)
        hidden = wrapped_model.init_hidden()
        outputs = wrapped_model.inference(x, hidden)
        self.output_dict = {key: np.zeros_like(value) for key, value in outputs.items()}

    def inference(self, *args):
        return self.output_dict


## agent

import random

import numpy as np


class RandomAgent:
    def reset(self, env, show=False):
        pass

    def action(self, env, player, show=False):
        actions = env.legal_actions(player)
        return random.choice(actions)

    def observe(self, env, player, show=False):
        return [0.0]


class RuleBasedAgent(RandomAgent):
    def action(self, env, player, show=False):
        if hasattr(env, 'rule_based_action'):
            return env.rule_based_action(player)
        else:
            return random.choice(env.legal_actions(player))


def print_outputs(env, prob, v):
    if hasattr(env, 'print_outputs'):
        env.print_outputs(prob, v)
    else:
        print('v = %f' % v)
        print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, observation=False, temperature=0.0):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
        self.observation = observation
        self.temperature = temperature

    def reset(self, env, show=False):
        self.hidden = self.model.init_hidden()

    def plan(self, obs):
        outputs = self.model.inference(obs, self.hidden)
        self.hidden = outputs.pop('hidden', None)
        return outputs

    def action(self, env, player, show=False):
        outputs = self.plan(env.observation(player))
        actions = env.legal_actions(player)
        p = outputs['policy']
        v = outputs.get('value', None)
        mask = np.ones_like(p)
        mask[actions] = 0
        p = p - mask * 1e32

        if show:
            print_outputs(env, softmax(p), v)

        if self.temperature == 0:
            ap_list = sorted([(a, p[a]) for a in actions], key=lambda x: -x[1])
            return ap_list[0][0]
        else:
            return random.choices(np.arange(len(p)), weights=softmax(p / self.temperature))[0]

    def observe(self, env, player, show=False):
        v = None
        if self.observation:
            outputs = self.plan(env.observation(player))
            v = outputs.get('value', None)
            if show:
                print_outputs(env, None, v)
        return v if v is not None else [0.0]


class EnsembleAgent(Agent):
    def reset(self, env, show=False):
        self.hidden = [model.init_hidden() for model in self.model]

    def plan(self, obs):
        outputs = {}
        for i, model in enumerate(self.model):
            o = model.inference(obs, self.hidden[i])
            for k, v in o:
                if k == 'hidden':
                    self.hidden[i] = v
                else:
                    outputs[k] = outputs.get(k, []) + [o]
        for k, vl in outputs:
            outputs[k] = np.mean(vl, axis=0)
        return outputs


class SoftAgent(Agent):
    def __init__(self, model, observation=False):
        super().__init__(model, observation=observation, temperature=1.0)


## generation

import random
import bz2
import pickle

import numpy as np


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, args):
        # episode generation
        moments = []
        hidden = {}
        for player in self.env.players():
            hidden[player] = models[player].init_hidden()

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            moment_keys = ['observation', 'policy', 'action_mask', 'action', 'value', 'reward', 'return']
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}

            turn_players = self.env.turns()
            for player in self.env.players():
                if player in turn_players or self.args['observation']:
                    obs = self.env.observation(player)
                    model = models[player]
                    outputs = model.inference(obs, hidden[player])
                    hidden[player] = outputs.get('hidden', None)
                    v = outputs.get('value', None)

                    moment['observation'][player] = obs
                    moment['value'][player] = v

                    if player in turn_players:
                        p_ = outputs['policy']
                        legal_actions = self.env.legal_actions(player)
                        action_mask = np.ones_like(p_) * 1e32
                        action_mask[legal_actions] = 0
                        p = p_ - action_mask
                        action = random.choices(legal_actions, weights=softmax(p[legal_actions]))[0]

                        moment['policy'][player] = p
                        moment['action_mask'][player] = action_mask
                        moment['action'][player] = action

            err = self.env.step(moment['action'])
            if err:
                return None

            reward = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = reward.get(player, None)

            moment['turn'] = turn_players
            moments.append(moment)

        if len(moments) < 1:
            return None

        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode


## evaluation

import random
import time
import multiprocessing as mp


network_match_port = 9876


def view(env, player=None):
    if hasattr(env, 'view'):
        env.view(player=player)
    else:
        print(env)


def view_transition(env):
    if hasattr(env, 'view_transition'):
        env.view_transition()
    else:
        pass


class NetworkAgentClient:
    def __init__(self, agent, env, conn):
        self.conn = conn
        self.agent = agent
        self.env = env

    def run(self):
        while True:
            command, args = self.conn.recv()
            if command == 'quit':
                break
            elif command == 'outcome':
                print('outcome = %f' % args[0])
            elif hasattr(self.agent, command):
                if command == 'action' or command == 'observe':
                    view(self.env)
                ret = getattr(self.agent, command)(self.env, *args, show=True)
                if command == 'action':
                    player = args[0]
                    ret = self.env.action2str(ret, player)
            else:
                ret = getattr(self.env, command)(*args)
                if command == 'update':
                    reset = args[1]
                    if reset:
                        self.agent.reset(self.env, show=True)
                    view_transition(self.env)
            self.conn.send(ret)


class NetworkAgent:
    def __init__(self, conn):
        self.conn = conn

    def update(self, data, reset):
        return send_recv(self.conn, ('update', [data, reset]))

    def outcome(self, outcome):
        return send_recv(self.conn, ('outcome', [outcome]))

    def action(self, player):
        return send_recv(self.conn, ('action', [player]))

    def observe(self, player):
        return send_recv(self.conn, ('observe', [player]))


def exec_match(env, agents, critic, show=False, game_args={}):
    ''' match with shared game environment '''
    if env.reset(game_args):
        return None
    for agent in agents.values():
        agent.reset(env, show=show)
    while not env.terminal():
        if show:
            view(env)
        if show and critic is not None:
            print('cv = ', critic.observe(env, None, show=False)[0])
        turn_players = env.turns()
        actions = {}
        for p, agent in agents.items():
            if p in turn_players:
                actions[p] = agent.action(env, p, show=show)
            else:
                agent.observe(env, p, show=show)
        if env.step(actions):
            return None
        if show:
            view_transition(env)
    outcome = env.outcome()
    if show:
        print('final outcome = %s' % outcome)
    return outcome


def exec_network_match(env, network_agents, critic, show=False, game_args={}):
    ''' match with divided game environment '''
    if env.reset(game_args):
        return None
    for p, agent in network_agents.items():
        info = env.diff_info(p)
        agent.update(info, True)
    while not env.terminal():
        if show:
            view(env)
        if show and critic is not None:
            print('cv = ', critic.observe(env, None, show=False)[0])
        turn_players = env.turns()
        actions = {}
        for p, agent in network_agents.items():
            if p in turn_players:
                action = agent.action(p)
                actions[p] = env.str2action(action, p)
            else:
                agent.observe(p)
        if env.step(actions):
            return None
        for p, agent in network_agents.items():
            info = env.diff_info(p)
            agent.update(info, False)
    outcome = env.outcome()
    for p, agent in network_agents.items():
        agent.outcome(outcome[p])
    return outcome


def build_agent(raw, env):
    if raw == 'random':
        return RandomAgent()
    elif raw == 'rulebase':
        return RuleBasedAgent()
    return None


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.default_opponent = 'random'

    def execute(self, models, args):
        opponents = self.args.get('eval', {}).get('opponent', [])
        if len(opponents) == 0:
            opponent = self.default_opponent
        else:
            opponent = random.choice(opponents)

        agents = {}
        for p, model in models.items():
            if model is None:
                agents[p] = build_agent(opponent, self.env)
            else:
                agents[p] = Agent(model, self.args['observation'])

        outcome = exec_match(self.env, agents, None)
        if outcome is None:
            print('None episode in evaluation!')
            return None
        return {'args': args, 'result': outcome, 'opponent': opponent}


def wp_func(results):
    games = sum([v for k, v in results.items() if k is not None])
    win = sum([(k + 1) / 2 * v for k, v in results.items() if k is not None])
    if games == 0:
        return 0.0
    return win / games


def eval_process_mp_child(agents, critic, env_args, index, in_queue, out_queue, seed, show=False):
    random.seed(seed + index)
    env = make_env({**env_args, 'id': index})
    while True:
        args = in_queue.get()
        if args is None:
            break
        g, agent_ids, pat_idx, game_args = args
        print('*** Game %d ***' % g)
        agent_map = {env.players()[p]: agents[ai] for p, ai in enumerate(agent_ids)}
        if isinstance(list(agent_map.values())[0], NetworkAgent):
            outcome = exec_network_match(env, agent_map, critic, show=show, game_args=game_args)
        else:
            outcome = exec_match(env, agent_map, critic, show=show, game_args=game_args)
        out_queue.put((pat_idx, agent_ids, outcome))
    out_queue.put(None)


def evaluate_mp(env, agents, critic, env_args, args_patterns, num_process, num_games, seed):
    in_queue, out_queue = mp.Queue(), mp.Queue()
    args_cnt = 0
    total_results, result_map = [{} for _ in agents], [{} for _ in agents]
    print('total games = %d' % (len(args_patterns) * num_games))
    time.sleep(0.1)
    for pat_idx, args in args_patterns.items():
        for i in range(num_games):
            if len(agents) == 2:
                # When playing two player game,
                # the number of games with first or second player is equalized.
                first_agent = 0 if i < (num_games + 1) // 2 else 1
                tmp_pat_idx, agent_ids = (pat_idx + '-F', [0, 1]) if first_agent == 0 else (pat_idx + '-S', [1, 0])
            else:
                tmp_pat_idx, agent_ids = pat_idx, random.sample(list(range(len(agents))), len(agents))
            in_queue.put((args_cnt, agent_ids, tmp_pat_idx, args))
            for p in range(len(agents)):
                result_map[p][tmp_pat_idx] = {}
            args_cnt += 1

    network_mode = agents[0] is None
    if network_mode:  # network battle mode
        agents = network_match_acception(num_process, env_args, len(agents), network_match_port)
    else:
        agents = [agents] * num_process

    for i in range(num_process):
        in_queue.put(None)
        args = agents[i], critic, env_args, i, in_queue, out_queue, seed
        if num_process > 1:
            mp.Process(target=eval_process_mp_child, args=args).start()
            if network_mode:
                for agent in agents[i]:
                    agent.conn.close()
        else:
            eval_process_mp_child(*args, show=True)

    finished_cnt = 0
    while finished_cnt < num_process:
        ret = out_queue.get()
        if ret is None:
            finished_cnt += 1
            continue
        pat_idx, agent_ids, outcome = ret
        if outcome is not None:
            for idx, p in enumerate(env.players()):
                agent_id = agent_ids[idx]
                oc = outcome[p]
                result_map[agent_id][pat_idx][oc] = result_map[agent_id][pat_idx].get(oc, 0) + 1
                total_results[agent_id][oc] = total_results[agent_id].get(oc, 0) + 1

    for p, r_map in enumerate(result_map):
        print('---agent %d---' % p)
        for pat_idx, results in r_map.items():
            print(pat_idx, {k: results[k] for k in sorted(results.keys(), reverse=True)}, wp_func(results))
        print('total', {k: total_results[p][k] for k in sorted(total_results[p].keys(), reverse=True)}, wp_func(total_results[p]))


def network_match_acception(n, env_args, num_agents, port):
    waiting_conns = []
    accepted_conns = []

    for conn in accept_socket_connections(port):
        if len(accepted_conns) >= n * num_agents:
            break
        waiting_conns.append(conn)

        if len(waiting_conns) == num_agents:
            conn = waiting_conns[0]
            accepted_conns.append(conn)
            waiting_conns = waiting_conns[1:]
            conn.send(env_args)  # send accept with environment arguments

    agents_list = [
        [NetworkAgent(accepted_conns[i * num_agents + j]) for j in range(num_agents)]
        for i in range(n)
    ]

    return agents_list


def get_model(env, model_path):
    import torch
    from .model import ModelWrapper
    model = env.net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return ModelWrapper(model)


def client_mp_child(env_args, model_path, conn):
    env = make_env(env_args)
    agent = build_agent(model_path, env)
    if agent is None:
        model = get_model(env, model_path)
        agent = Agent(model)
    NetworkAgentClient(agent, env, conn).run()


def eval_main(args, argv):
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    model_path = argv[0] if len(argv) >= 1 else 'models/latest.pth'
    num_games = int(argv[1]) if len(argv) >= 2 else 100
    num_process = int(argv[2]) if len(argv) >= 3 else 1

    agent1 = build_agent(model_path, env)
    if agent1 is None:
        agent1 = Agent(get_model(env, model_path))
    critic = None

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    agents = [agent1] + [RandomAgent() for _ in range(len(env.players()) - 1)]

    evaluate_mp(env, agents, critic, env_args, {'default': {}}, num_process, num_games, seed)


def eval_server_main(args, argv):
    print('network match server mode')
    env_args = args['env_args']
    prepare_env(env_args)
    env = make_env(env_args)

    num_games = int(argv[0]) if len(argv) >= 1 else 100
    num_process = int(argv[1]) if len(argv) >= 2 else 1

    print('%d process, %d games' % (num_process, num_games))

    seed = random.randrange(1e8)
    print('seed = %d' % seed)

    evaluate_mp(env, [None] * len(env.players()), None, env_args, {'default': {}}, num_process, num_games, seed)


def eval_client_main(args, argv):
    print('network match client mode')
    while True:
        try:
            host = argv[1] if len(argv) >= 2 else 'localhost'
            conn = connect_socket_connection(host, network_match_port)
            env_args = conn.recv()
        except EOFError:
            break

        model_path = argv[0] if len(argv) >= 1 else 'models/latest.pth'
        mp.Process(target=client_mp_child, args=(env_args, model_path, conn)).start()
        conn.close()


## worker

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


def make_worker_args(args, n_ga, gaid, wid, conn):
    return args, conn, wid * n_ga + gaid


def open_worker(args, conn, wid):
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    def __init__(self, args, conn, gaid):
        print('started gather %d' % gaid)
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque([])
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_parallel'], args['worker']['num_gathers']

        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid)
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


def gather_loop(args, conn, gaid):
    try:
        gather = Gather(args, conn, gaid)
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

    def run(self):
        # prepare listening connections
        def entry_server(port):
            print('started entry server %d' % port)
            conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
            while not self.shutdown_flag:
                conn = next(conn_acceptor)
                if conn is not None:
                    worker_args = conn.recv()
                    print('accepted connection from %s!' % worker_args['address'])
                    args = copy.deepcopy(self.args)
                    args['worker'] = worker_args
                    conn.send(args)
                    conn.close()
            print('finished entry server')

        def worker_server(port):
            conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
            print('started worker server %d' % port)
            while not self.shutdown_flag:  # use super class's flag
                conn = next(conn_acceptor)
                if conn is not None:
                    self.add_connection(conn)
            print('finished worker server')

        # use thread list of super class
        self.threads.append(threading.Thread(target=entry_server, args=(9999,)))
        self.threads.append(threading.Thread(target=worker_server, args=(9998,)))
        self.threads[-2].start()
        self.threads[-1].start()


def entry(worker_args):
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


class RemoteWorkerCluster:
    def __init__(self, args):
        args['address'] = gethostname()
        if 'num_gathers' not in args:
            args['num_gathers'] = 1 + max(0, args['num_parallel'] - 1) // 16

        self.args = args

    def run(self):
        args = entry(self.args)
        print(args)
        prepare_env(args['env'])

        # open worker
        process = []
        try:
            for i in range(self.args['num_gathers']):
                conn = connect_socket_connection(self.args['server_address'], 9998)
                p = mp.Process(target=gather_loop, args=(args, conn, i))
                p.start()
                conn.close()
                process.append(p)
            while True:
                time.sleep(100)
        finally:
            for p in process:
                p.terminate()


def worker_main(args):
    # offline generation worker
    worker = RemoteWorkerCluster(args=args['worker_args'])
    worker.run()


## losses

from collections import deque

import torch


def monte_carlo(values, returns):
    return returns, returns - values


def temporal_difference(values, returns, rewards, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        reward = rewards[:, i] if rewards is not None else 0
        target_values.appendleft(reward + gamma * ((1 - lmb) * values[:, i + 1] + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return target_values, target_values - values


def upgo(values, returns, rewards, lmb, gamma):
    target_values = deque([returns[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        value = values[:, i + 1]
        reward = rewards[:, i] if rewards is not None else 0
        target_values.appendleft(reward + gamma * torch.max(value, (1 - lmb) * value + lmb * target_values[0]))

    target_values = torch.stack(tuple(target_values), dim=1)

    return target_values, target_values - values


def vtrace(values, returns, rewards, lmb, gamma, rhos, cs):
    rewards = rewards if rewards is not None else 0
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1)
    deltas = rhos * (rewards + gamma * values_t_plus_1 - values)

    # compute Vtrace value target recursively
    vs_minus_v_xs = deque([deltas[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[:, i] + gamma * lmb * cs[:, i] * vs_minus_v_xs[0])

    vs_minus_v_xs = torch.stack(tuple(vs_minus_v_xs), dim=1)
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[:, 1:], returns[:, -1:]], dim=1)
    advantages = rewards + gamma * vs_t_plus_1 - values

    return vs, advantages


def compute_target(algorithm, values, returns, rewards, lmb, gamma, rhos, cs):
    if values is None:
        # In the absence of a baseline, Monte Carlo returns are used.
        return returns, returns

    if algorithm == 'MC':
        return monte_carlo(values, returns)
    elif algorithm == 'TD':
        return temporal_difference(values, returns, rewards, lmb, gamma)
    elif algorithm == 'UPGO':
        return upgo(values, returns, rewards, lmb, gamma)
    elif algorithm == 'VTRACE':
        return vtrace(values, returns, rewards, lmb, gamma, rhos, cs)
    else:
        print('No algorithm named %s' % algorithm)


## train

import os
import time
import copy
import threading
import random
import bz2
import pickle
import warnings
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import psutil


def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (B, T, P, ...) .
        (B is batch size, T is time length, P is player count)
    """

    obss, datum = [], []

    def replace_none(a, b):
        return a if a is not None else b

    for ep in episodes:
        moments_ = sum([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        moments = moments_[ep['start'] - ep['base']:ep['end'] - ep['base']]
        players = list(moments[0]['observation'].keys())
        if not args['turn_based_training']:  # solo training
            players = [random.choice(players)]

        obs_zeros = map_r(moments[0]['observation'][moments[0]['turn'][0]], lambda o: np.zeros_like(o))  # template for padding
        p_zeros = np.zeros_like(moments[0]['policy'][moments[0]['turn'][0]])  # template for padding

        # data that is chainge by training configuration
        if args['turn_based_training'] and not args['observation']:
            obs = [[m['observation'][m['turn'][0]]] for m in moments]
            p = np.array([[m['policy'][m['turn'][0]]] for m in moments])
            act = np.array([[m['action'][m['turn'][0]]] for m in moments], dtype=np.int64)[..., np.newaxis]
            amask = np.array([[m['action_mask'][m['turn'][0]]] for m in moments])
        else:
            obs = [[replace_none(m['observation'][player], obs_zeros) for player in players] for m in moments]
            p = np.array([[replace_none(m['policy'][player], p_zeros) for player in players] for m in moments])
            act = np.array([[replace_none(m['action'][player], 0) for player in players] for m in moments], dtype=np.int64)[..., np.newaxis]
            amask = np.array([[replace_none(m['action_mask'][player], p_zeros + 1e32) for player in players] for m in moments])

        # reshape observation
        obs = rotate(rotate(obs))  # (T, P, ..., ...) -> (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o))

        # datum that is not changed by training configuration
        v = np.array([[replace_none(m['value'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        rew = np.array([[replace_none(m['reward'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        ret = np.array([[replace_none(m['return'][player], [0]) for player in players] for m in moments], dtype=np.float32).reshape(len(moments), len(players), -1)
        oc = np.array([ep['outcome'][player] for player in players], dtype=np.float32).reshape(1, len(players), -1)

        emask = np.ones((len(moments), 1, 1), dtype=np.float32)  # episode mask
        tmask = np.array([[[m['policy'][player] is not None] for player in players] for m in moments], dtype=np.float32)
        omask = np.array([[[m['value'][player] is not None] for player in players] for m in moments], dtype=np.float32)

        progress = np.arange(ep['start'], ep['end'], dtype=np.float32)[..., np.newaxis] / ep['total']

        # pad each array if step length is short
        if len(tmask) < args['forward_steps']:
            pad_len = args['forward_steps'] - len(tmask)
            obs = map_r(obs, lambda o: np.pad(o, [(0, pad_len)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            p = np.pad(p, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            v = np.concatenate([v, np.tile(oc, [pad_len, 1, 1])])
            act = np.pad(act, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            rew = np.pad(rew, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            ret = np.pad(ret, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            emask = np.pad(emask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            tmask = np.pad(tmask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            omask = np.pad(omask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            amask = np.pad(amask, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1e32)
            progress = np.pad(progress, [(0, pad_len), (0, 0)], 'constant', constant_values=1)

        obss.append(obs)
        datum.append((p, v, act, oc, rew, ret, emask, tmask, omask, amask, progress))

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    p, v, act, oc, rew, ret, emask, tmask, omask, amask, progress = [to_torch(np.array(val)) for val in zip(*datum)]

    return {
        'observation': obs,
        'policy': p, 'value': v,
        'action': act, 'outcome': oc,
        'reward': rew, 'return': ret,
        'episode_mask': emask,
        'turn_mask': tmask, 'observation_mask': omask,
        'action_mask': amask,
        'progress': progress,
    }


def forward_prediction(model, hidden, batch, args):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: batch outputs of neural network
    """

    observations = batch['observation']  # (B, T, P, ...)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.view(-1, *o.size()[3:]))
        outputs = model(obs, None)
    else:
        # sequential computation with RNN
        outputs = {}
        for t in range(batch['turn_mask'].size(1)):
            obs = map_r(observations, lambda o: o[:, t].reshape(-1, *o.size()[3:]))  # (..., B * P, ...)
            omask_ = batch['observation_mask'][:, t]
            omask = map_r(hidden, lambda h: omask_.view(*h.size()[:2], *([1] * (len(h.size()) - 2))))
            hidden_ = bimap_r(hidden, omask, lambda h, m: h * m)  # (..., B, P, ...)
            if args['turn_based_training'] and not args['observation']:
                hidden_ = map_r(hidden_, lambda h: h.sum(1))  # (..., B * 1, ...)
            else:
                hidden_ = map_r(hidden_, lambda h: h.view(-1, *h.size()[2:]))  # (..., B * P, ...)
            outputs_ = model(obs, hidden_)
            for k, o in outputs_.items():
                if k == 'hidden':
                    next_hidden = outputs_['hidden']
                else:
                    outputs[k] = outputs.get(k, []) + [o]
            next_hidden = bimap_r(next_hidden, hidden, lambda nh, h: nh.view(h.size(0), -1, *h.size()[2:]))  # (..., B, P or 1, ...)
            hidden = trimap_r(hidden, next_hidden, omask, lambda h, nh, m: h * (1 - m) + nh * m)
        outputs = {k: torch.stack(o, dim=1) for k, o in outputs.items() if o[0] is not None}

    for k, o in outputs.items():
        o = o.view(*batch['turn_mask'].size()[:2], -1, o.size(-1))
        if k == 'policy':
            # gather turn player's policies
            outputs[k] = o.mul(batch['turn_mask']).sum(2, keepdim=True) - batch['action_mask']
        else:
            # mask valid target values and cumulative rewards
            outputs[k] = o.mul(batch['observation_mask'])

    return outputs


def compose_losses(outputs, log_selected_policies, total_advantages, targets, batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    tmasks = batch['turn_mask']
    omasks = batch['observation_mask']

    losses = {}
    dcnt = tmasks.sum().item()
    turn_advantages = total_advantages.mul(tmasks).sum(2, keepdim=True)

    losses['p'] = (-log_selected_policies * turn_advantages).sum()
    if 'value' in outputs:
        losses['v'] = ((outputs['value'] - targets['value']) ** 2).mul(omasks).sum() / 2
    if 'return' in outputs:
        losses['r'] = F.smooth_l1_loss(outputs['return'], targets['return'], reduction='none').mul(omasks).sum()

    entropy = dist.Categorical(logits=outputs['policy']).entropy().mul(tmasks.sum(-1))
    losses['ent'] = entropy.sum()

    base_loss = losses['p'] + losses.get('v', 0) + losses.get('r', 0)
    entropy_loss = entropy.mul(1 - batch['progress'] * (1 - args['entropy_regularization_decay'])).sum() * -args['entropy_regularization']
    losses['total'] = base_loss + entropy_loss

    return losses, dcnt


def compute_loss(batch, model, hidden, args):
    outputs = forward_prediction(model, hidden, batch, args)
    actions = batch['action']
    emasks = batch['episode_mask']
    clip_rho_threshold, clip_c_threshold = 1.0, 1.0

    log_selected_b_policies = F.log_softmax(batch['policy']  , dim=-1).gather(-1, actions) * emasks
    log_selected_t_policies = F.log_softmax(outputs['policy'], dim=-1).gather(-1, actions) * emasks

    # thresholds of importance sampling
    log_rhos = log_selected_t_policies.detach() - log_selected_b_policies
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    outputs_nograd = {k: o.detach() for k, o in outputs.items()}

    if 'value' in outputs_nograd:
        values_nograd = outputs_nograd['value']
        if args['turn_based_training'] and values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            values_nograd = (values_nograd + values_nograd_opponent) / (batch['observation_mask'].sum(dim=2, keepdim=True) + 1e-8)
        outputs_nograd['value'] = values_nograd * emasks + batch['outcome'] * (1 - emasks)

    # compute targets and advantage
    targets = {}
    advantages = {}

    value_args = outputs_nograd.get('value', None), batch['outcome'], None, args['lambda'], 1, clipped_rhos, cs
    return_args = outputs_nograd.get('return', None), batch['return'], batch['reward'], args['lambda'], args['gamma'], clipped_rhos, cs

    targets['value'], advantages['value'] = compute_target(args['value_target'], *value_args)
    targets['return'], advantages['return'] = compute_target(args['value_target'], *return_args)

    if args['policy_target'] != args['value_target']:
        _, advantages['value'] = compute_target(args['policy_target'], *value_args)
        _, advantages['return'] = compute_target(args['policy_target'], *return_args)

    # compute policy advantage
    total_advantages = clipped_rhos * sum(advantages.values())

    return compose_losses(outputs, log_selected_t_policies, total_advantages, targets, batch, args)


class Batcher:
    def __init__(self, args, episodes):
        self.args = args
        self.episodes = episodes
        self.shutdown_flag = False

        self.executor = MultiProcessJobExecutor(self._worker, self._selector(), self.args['num_batchers'], num_receivers=2)

    def _selector(self):
        while True:
            yield [self.select_episode() for _ in range(self.args['batch_size'])]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while not self.shutdown_flag:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args)
            conn.send(batch)
        print('finished batcher %d' % bid)

    def run(self):
        self.executor.start()

    def select_episode(self):
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep['steps'] - self.args['forward_steps'])  # change start turn by sequence length
        st = random.randrange(turn_candidates)
        ed = min(st + self.args['forward_steps'], ep['steps'])
        st_block = st // self.args['compress_steps']
        ed_block = (ed - 1) // self.args['compress_steps'] + 1
        ep_minimum = {
            'args': ep['args'], 'outcome': ep['outcome'],
            'moment': ep['moment'][st_block:ed_block],
            'base': st_block * self.args['compress_steps'],
            'start': st, 'end': ed, 'total': ep['steps'],
        }
        return ep_minimum

    def batch(self):
        return self.executor.recv()

    def shutdown(self):
        self.shutdown_flag = True
        self.executor.shutdown()


class Trainer:
    def __init__(self, args, model):
        self.episodes = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.default_lr = 3e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.default_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        self.lock = threading.Lock()
        self.batcher = Batcher(self.args, self.episodes)
        self.updated_model = None, 0
        self.update_flag = False
        self.shutdown_flag = False

        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model
        if self.gpu > 1:
            self.trained_model = nn.DataParallel(self.wrapped_model)

    def update(self):
        if len(self.episodes) < self.args['minimum_episodes']:
            return None, 0  # return None before training
        self.update_flag = True
        while True:
            time.sleep(0.1)
            model, steps = self.recheck_update()
            if model is not None:
                break
        return model, steps

    def report_update(self, model, steps):
        self.lock.acquire()
        self.update_flag = False
        self.updated_model = model, steps
        self.lock.release()

    def recheck_update(self):
        self.lock.acquire()
        flag = self.update_flag
        self.lock.release()
        return (None, -1) if flag else self.updated_model

    def shutdown(self):
        self.shutdown_flag = True
        self.batcher.shutdown()

    def train(self):
        if self.optimizer is None:  # non-parametric model
            print()
            return

        batch_cnt, data_cnt, loss_sum = 0, 0, {}
        if self.gpu > 0:
            self.trained_model.cuda()
        self.trained_model.train()

        while data_cnt == 0 or not (self.update_flag or self.shutdown_flag):
            batch = self.batcher.batch()
            batch_size = batch['value'].size(0)
            player_count = batch['value'].size(2)
            hidden = self.wrapped_model.init_hidden([batch_size, player_count])
            if self.gpu > 0:
                batch = to_gpu(batch)
                hidden = to_gpu(hidden)

            losses, dcnt = compute_loss(batch, self.trained_model, hidden, self.args)

            self.optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            batch_cnt += 1
            data_cnt += dcnt
            for k, l in losses.items():
                loss_sum[k] = loss_sum.get(k, 0.0) + l.item()

            self.steps += 1

        print('loss = %s' % ' '.join([k + ':' + '%.3f' % (l / data_cnt) for k, l in loss_sum.items()]))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.default_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        self.model.cpu()
        self.model.eval()
        return copy.deepcopy(self.model)

    def run(self):
        print('waiting training')
        while not self.shutdown_flag:
            if len(self.episodes) < self.args['minimum_episodes']:
                time.sleep(1)
                continue
            if self.steps == 0:
                self.batcher.run()
                print('started training')
            model = self.train()
            self.report_update(model, self.steps)
        print('finished training')


class Learner:
    def __init__(self, args, net=None, remote=False):
        train_args = args['train_args']
        env_args = args['env_args']
        train_args['env'] = env_args
        args = train_args

        self.args = args
        random.seed(args['seed'])

        self.env = make_env(env_args)
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.shutdown_flag = False
        self.flags = set()

        # trained datum
        self.model_epoch = self.args['restart_epoch']
        self.model = net if net is not None else self.env.net()
        if self.model_epoch > 0:
            self.model.load_state_dict(torch.load(self.model_path(self.model_epoch)), strict=False)

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0

        # multiprocess or remote connection
        self.worker = WorkerServer(args) if remote else WorkerCluster(args)

        # thread connection
        self.trainer = Trainer(args, self.model)

    def shutdown(self):
        self.shutdown_flag = True
        self.trainer.shutdown()
        self.worker.shutdown()
        self.thread.join()

    def model_path(self, model_id):
        return os.path.join('models', str(model_id) + '.pth')

    def latest_model_path(self):
        return os.path.join('models', 'latest.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_epoch += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), self.model_path(self.model_epoch))
        torch.save(model.state_dict(), self.latest_model_path())

    def feed_episodes(self, episodes):
        # analyze generated episodes
        for episode in episodes:
            if episode is None:
                continue
            for p in episode['args']['player']:
                model_id = episode['args']['model_id'][p]
                outcome = episode['outcome'][p]
                n, r, r2 = self.generation_results.get(model_id, (0, 0, 0))
                self.generation_results[model_id] = n + 1, r + outcome, r2 + outcome ** 2

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])

        mem_percent = psutil.virtual_memory().percent
        mem_ok = mem_percent <= 95
        maximum_episodes = self.args['maximum_episodes'] if mem_ok else int(len(self.trainer.episodes) * 95 / mem_percent)

        if not mem_ok and 'memory_over' not in self.flags:
            warnings.warn("memory usage %.1f%% with buffer size %d" % (mem_percent, len(self.trainer.episodes)))
            self.flags.add('memory_over')

        while len(self.trainer.episodes) > maximum_episodes:
            self.trainer.episodes.popleft()

    def feed_results(self, results):
        # store evaluation results
        for result in results:
            if result is None:
                continue
            for p in result['args']['player']:
                model_id = result['args']['model_id'][p]
                res = result['result'][p]
                n, r, r2 = self.results.get(model_id, (0, 0, 0))
                self.results[model_id] = n + 1, r + res, r2 + res ** 2

                if model_id not in self.results_per_opponent:
                    self.results_per_opponent[model_id] = {}
                opponent = result['opponent']
                n, r, r2 = self.results_per_opponent[model_id].get(opponent, (0, 0, 0))
                self.results_per_opponent[model_id][opponent] = n + 1, r + res, r2 + res ** 2

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_epoch)

        if self.model_epoch not in self.results:
            print('win rate = Nan (0)')
        else:
            def output_wp(name, results):
                n, r, r2 = results
                mean = r / (n + 1e-6)
                name_tag = ' (%s)' % name if name != '' else ''
                print('win rate%s = %.3f (%.1f / %d)' % (name_tag, (mean + 1) / 2, (r + n) / 2, n))

            if len(self.args.get('eval', {}).get('opponent', [])) <= 1:
                output_wp('', self.results[self.model_epoch])
            else:
                output_wp('total', self.results[self.model_epoch])
                for key in sorted(list(self.results_per_opponent[self.model_epoch])):
                    output_wp(key, self.results_per_opponent[self.model_epoch][key])

        if self.model_epoch not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_epoch]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))

        model, steps = self.trainer.update()
        if model is None:
            model = self.model
        self.update_model(model, steps)

        # clear flags
        self.flags = set()

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        while self.model_epoch < self.args['epochs'] or self.args['epochs'] < 0:
            # no update call before storing minimum number of episodes + 1 age
            next_update_episodes = prev_update_episodes + self.args['update_episodes']
            while not self.shutdown_flag and self.num_episodes < next_update_episodes:
                conn, (req, data) = self.worker.recv()
                multi_req = isinstance(data, list)
                if not multi_req:
                    data = [data]
                send_data = []

                if req == 'args':
                    for _ in data:
                        args = {'model_id': {}}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        else:
                            args['role'] = 'g'

                        if args['role'] == 'g':
                            # genatation configuration
                            args['player'] = self.env.players()
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                else:
                                    args['model_id'][p] = -1
                            self.num_episodes += 1
                            if self.num_episodes % 100 == 0:
                                print(self.num_episodes, end=' ', flush=True)

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = [self.env.players()[self.num_results % len(self.env.players())]]
                            for p in self.env.players():
                                if p in args['player']:
                                    args['model_id'][p] = self.model_epoch
                                else:
                                    args['model_id'][p] = -1
                            self.num_results += 1

                        send_data.append(args)

                elif req == 'episode':
                    # report generated episodes
                    self.feed_episodes(data)
                    send_data = [None] * len(data)

                elif req == 'result':
                    # report evaluation results
                    self.feed_results(data)
                    send_data = [None] * len(data)

                elif req == 'model':
                    for model_id in data:
                        model = self.model
                        if model_id != self.model_epoch and model_id > 0:
                            try:
                                model = copy.deepcopy(self.model)
                                model.load_state_dict(torch.load(self.model_path(model_id)), strict=False)
                            except:
                                # return latest model if failed to load specified model
                                pass
                        send_data.append(pickle.dumps(model))

                if not multi_req and len(send_data) == 1:
                    send_data = send_data[0]
                self.worker.send(conn, send_data)
            prev_update_episodes = next_update_episodes
            self.update()
        print('finished server')

    def run(self):
        try:
            # open training thread
            self.thread = threading.Thread(target=self.trainer.run)
            self.thread.start()
            # open generator, evaluator
            self.worker.run()
            self.server()

        finally:
            self.shutdown()


def train_main(args):
    prepare_env(args['env_args'])  # preparing environment is needed in stand-alone mode
    learner = Learner(args=args)
    learner.run()


def train_server_main(args):
    learner = Learner(args=args, remote=True)
    learner.run()
