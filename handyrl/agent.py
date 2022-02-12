# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# agent classes

import random

import numpy as np

from .util import softmax


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
        if v is not None:
            print('v = %f' % v)
        if prob is not None:
            print('p = %s' % (prob * 1000).astype(int))


class Agent:
    def __init__(self, model, temperature=0.0):
        # model might be a neural net, or some planning algorithm such as game tree search
        self.model = model
        self.hidden = None
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
    def __init__(self, model):
        super().__init__(model, temperature=1.0)


# model

class OnnxModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.ort_session = None

    def _open_session(self):
        import os
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.ort_session = onnxruntime.InferenceSession(self.model_path, sess_options=opts)

    def init_hidden(self):
        if self.ort_session is None:
            self._open_session()
        hidden_inputs = [y for y in self.ort_session.get_inputs() if y.name.startswith('hidden')]
        if len(hidden_inputs) == 0:
            return None
        type_map = {
            'tensor(float)': np.float32,
            'tensor(int64)': np.int64,
        }
        hidden_tensors = [np.zeros(y.shape[1:], dtype=type_map[y.type]) for y in hidden_inputs]
        return hidden_tensors

    def inference(self, x, hidden=None, batch_input=False):
        # numpy array -> numpy array
        if self.ort_session is None:
            self._open_session()

        ort_inputs = {}
        ort_input_names = [y.name for y in self.ort_session.get_inputs()]

        def insert_input(y):
            y = y if batch_input else np.expand_dims(y, 0)
            ort_inputs[ort_input_names[len(ort_inputs)]] = y
        from .util import map_r
        map_r(x, lambda y: insert_input(y))
        if hidden is not None:
            map_r(hidden, lambda y: insert_input(y))
        ort_outputs = self.ort_session.run(None, ort_inputs)
        if not batch_input:
            ort_outputs = [o.squeeze(0) for o in ort_outputs]

        ort_output_names = [y.name for y in self.ort_session.get_outputs()]
        outputs = {name: ort_outputs[i] for i, name in enumerate(ort_output_names)}

        hidden_outputs = []
        for k in list(outputs.keys()):
            if k.startswith('hidden'):
                hidden_outputs.append(outputs.pop(k))
        if len(hidden_outputs) == 0:
            hidden_outputs = None

        outputs = {**outputs, 'hidden': hidden_outputs}
        return outputs


def load_model(model_path, model):
    if model_path.endswith('.onnx'):
        model = OnnxModel(model_path)
        return model
    import torch
    from .model import ModelWrapper
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return ModelWrapper(model)
