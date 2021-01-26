# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# tree search

import copy
import time

import numpy as np


def softmax(x):
    x = np.exp(x - np.max(x, axis=-1))
    return x / x.sum(axis=-1)


class Node:
    '''Search result of one abstract (or root) state'''
    def __init__(self, p, v):
        self.p, self.v = p, v
        self.n = np.zeros_like(p)
        self.q_sum = np.zeros((*p.shape, 2))
        self.n_all, self.q_sum_all = 1, v / 2  # prior

    def update(self, action, q_new):
        # Update
        self.n[action] += 1
        self.q_sum[action] += q_new

        # Update overall stats
        self.n_all += 1
        self.q_sum_all += q_new


class Edge:
    '''Search result of trainsition'''
    def __init__(self, q):
        self.n = {}
        self.v_sum = {}
        self.n_all, self.v_sum_all = 1, q if q is not None else 0  # prior

    def update(self, pattern, v_new):
        # Update
        self.n[pattern] = self.n.get(pattern, 0) + 1
        self.v_sum[pattern] = self.v_sum.get(pattern, np.zeros_like(v_new)) + v_new

        # Update overall stats
        self.n_all += 1
        self.v_sum_all += v_new


class MonteCarloTree:
    '''Monte Carlo Tree Search'''
    def __init__(self, model, args):
        self.model = model
        self.nodes = {}
        self.edges = {}
        self.args = {}

    def search(self, rp, path):
        # Return predicted value from new state
        key = '|' + ''.join(path)
        if key not in self.nodes:
            p, v = self.model['prediction'].inference(rp)
            p, v = softmax(p), v
            self.nodes[key] = Node(p, v)
            return v

        # Choose action with bandit
        node = self.nodes[key]
        p = node.p
        if len(path) == 0:
            # Add noise to policy on the root node
            p = 0.75 * p + 0.25 * np.random.dirichlet([0.15] * np.prod(p.shape)).reshape(*p.shape)
            # On the root node, we choose action only from legal actions
            p /= p.sum() + 1e-16

        q_mean_all = node.q_sum_all.reshape(1, -1) / node.n_all
        n, q_sum = 1 + node.n, q_mean_all + node.q_sum
        adv = (q_sum / n.reshape(-1, 1) - q_mean_all).reshape(2, -1, 2)
        adv = np.concatenate([adv[0, :, 0], adv[1, :, 1]])
        ucb = adv + 2.0 * np.sqrt(node.n_all) * p / n  # PUCB formula
        selected_action = np.argmax(ucb)

        # Search next state by recursively calling this function
        path.append('a' + str(selected_action))
        q_new = self.transition(rp, path, selected_action)
        node.update(selected_action, q_new)

        return q_new

    def transition(self, rp, path, action, q=None):
        key = '+' + ''.join(path)
        if key not in self.edges:
            self.edges[key] = Edge(q)

        # Choose next state with double progressive widening
        edge = self.edges[key]
        k = 1.5 * (edge.n_all ** 0.3)
        if k >= len(edge.n) and len(edge.n) < 16:
            unused_pattens = [pattern for pattern in range(16) if pattern not in edge.n]
            selected_pattern = np.random.choice(unused_pattens)
        else:
            weights = np.array([n for n in edge.n.values()]) / sum(edge.n.values())
            selected_pattern = np.random.choice(list(edge.n.keys()), p=weights)

        # State transition with selected action and pattern
        path.append('s' + str(selected_pattern))
        next_rp = self.model['dynamics'].inference(rp, np.array([action]), np.array([[selected_pattern]]))[0]
        v_new = self.search(next_rp, path)
        edge.update(selected_pattern, v_new)

        return v_new

    def think(self, root_obs, num_simulations, env=None, show=False):
        # End point of MCTS
        start, prev_time = time.time(), 0
        for _ in range(num_simulations):
            self.search(self.model['representation'].inference(root_obs), [])

            # Display search result on every second
            if show:
                tmp_time = time.time() - start
                if int(tmp_time) > int(prev_time):
                    prev_time = tmp_time
                    root, pv = self.nodes['|'], self.pv(env)
                    print('%.2f sec. best %s. q = %.4f. n = %d / %d. pv = %s'
                          % (tmp_time, env.action2str(pv[0][0], pv[0][1]), root.q_sum[pv[0][0]] / root.n[pv[0][0]],
                             root.n[pv[0][0]], root.n_all, ' '.join([env.action2str(a, p) for a, p in pv])))

        #  Return probability distribution weighted by the number of simulations
        root = self.nodes['|']
        n = root.n + 0.1
        p = np.log(n / n.sum())
        v = (root.q_sum * p).sum()
        return p, v

    def pv(self, env_):
        # Return principal variation (action sequence which is considered as the best)
        env = copy.deepcopy(env_)
        pv_seq = []
        while True:
            path = list(zip(*pv_seq))[0]
            key = '|' + ' '.join(map(str, path))
            if key not in self.nodes or self.nodes[key].n.sum() == 0:
                break
            best_action = sorted([(a, self.nodes[key].n[a]) for a in env.legal_actions()], key=lambda x: -x[1])[0][0]
            pv_seq.append((best_action, env.turn()))
            env.play(best_action)
        return pv_seq
