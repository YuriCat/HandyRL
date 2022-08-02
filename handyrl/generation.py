# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from .util import softmax


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, args):
        # episode generation
        moments = {}
        hidden = {}
        for player in self.env.players():
            moments[player] = []
            hidden[player] = models[player].init_hidden()

        err = self.env.reset()
        if err:
            return None

        while not self.env.terminal():
            moment_keys = ['observation', 'selected_prob', 'action_mask', 'action', 'value', 'reward', 'return']
            moment = {player: {key: None for key in moment_keys} for player in self.env.players()}
            actions = {}

            turn_players = self.env.turns()
            observers = self.env.observers()
            for player in self.env.players():
                if player not in turn_players + observers:
                    continue
                if player not in turn_players and player in args['player'] and not self.args['observation']:
                    continue

                obs = self.env.observation(player)
                model = models[player]
                outputs = model.inference(obs, hidden[player])
                hidden[player] = outputs.get('hidden', None)
                v = outputs.get('value', None)

                moment[player]['observation'] = obs
                moment[player]['value'] = v

                if player in turn_players:
                    p_ = outputs['policy']
                    legal_actions = self.env.legal_actions(player)
                    action_mask = np.ones_like(p_) * 1e32
                    action_mask[legal_actions] = 0
                    p = softmax(p_ - action_mask)
                    action = random.choices(legal_actions, weights=p[legal_actions])[0]

                    moment[player]['selected_prob'] = p[action]
                    moment[player]['action_mask'] = action_mask
                    moment[player]['action'] = action
                    actions[player] = action

            err = self.env.step(actions)
            if err:
                return None

            reward = self.env.reward()
            for player in self.env.players():
                moment[player]['reward'] = reward.get(player, None)

            for player, m in moment.items():
                moments[player].append(moment[player])

        if len(moments) < 1:
            return None

        for moment in moments.values():
            ret = 0
            for i, m in reversed(list(enumerate(moment))):
                ret = (m['reward'] or 0) + self.args['gamma'] * ret
                moment[i]['return'] = ret

        episode = {
            'args': args, 'steps': len(moments),
            'outcome': self.env.outcome(),
            'moment': {
                player: [
                    bz2.compress(pickle.dumps(moments_[i:i+self.args['compress_steps']]))
                    for i in range(0, len(moments_), self.args['compress_steps'])
                ] for player, moments_ in moments.items()
            }
        }

        return episode

    def execute(self, models, args):
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
