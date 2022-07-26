# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation from records

import random
import bz2
import pickle
import json
import glob

import numpy as np

from .util import softmax


class Replayer:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def replay(self, record, args):
        # episode replay
        moments = []

        err = self.env.update(None, reset=True)
        if err:
            return None

        for action in record.split(' '):
            moment_keys = ['observation', 'selected_prob', 'action_mask', 'action', 'value', 'reward', 'return']
            moment = {key: {p: None for p in self.env.players()} for key in moment_keys}

            turn_players = self.env.turns()
            observers = self.env.observers()
            for player in self.env.players():
                if player not in turn_players + observers:
                    continue
                if player not in turn_players and player in args['player'] and not self.args['observation']:
                    continue

                obs = self.env.observation(player)
                moment['observation'][player] = self.env.observation(player)

                if player in turn_players:
                    legal_actions = self.env.legal_actions(player)
                    action_mask = np.ones(9, dtype=np.float32) * 1e32
                    action_mask[legal_actions] = 0

                    moment['selected_prob'][player] = 1.0
                    moment['action_mask'][player] = action_mask
                    moment['action'][player] = self.env.str2action(action, player)

                moment['turn'] = turn_players
                self.env.update(action, reset=False)
                moments.append(moment)

        if len(moments) < 1:
            return None

        replay = {
            'replay': True,
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': [
                bz2.compress(pickle.dumps(moments[i:i+self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ]
        }

        return replay

    def _select_record(self):
        return 'B2 A1 A3 C1 B1 B3 C2 A2 C3'

    def execute(self, args):
        record = self._select_record()
        replay = self.replay(record, args)
        if replay is None:
            print('None episode in replay!')
        return replay
