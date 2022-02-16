# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# neural nets

import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn as nn
import torch.nn.functional as F

from .util import map_r


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
            if batch_size is None:  # for inference
                hidden = self.model.init_hidden([])
                return map_r(hidden, lambda h: h.detach().numpy() if isinstance(h, torch.Tensor) else h)
            else:  # for training
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


# GBDT

class BoostingModel:
    def __init__(self, action_length):
        self.action_length = action_length
        self.actor, self.critic = None, None

    def prepare(self, dmats):
        if self.actor is not None:
            return
        from xgboost import Booster
        xgb_p, xgb_wp = dmats
        basic_params = {'n_jobs': 1, 'learning_rate': 1e-6, 'booster': 'dart', 'rate_drop': 3e-2}
        self.actor = Booster({'objective': 'multi:softprob', 'num_class': self.action_length, **basic_params}, [xgb_p])
        self.critic = Booster({'objective': 'binary:logistic', **basic_params}, [xgb_wp])

    def cuda(self):
        if self.actor is not None:
            self.actor.set_param('tree_method', 'gpu_hist')
            self.critic.set_param('tree_method', 'gpu_hist')

    def cpu(self):
        if self.actor is not None:
            self.actor.set_param('tree_method', 'hist')
            self.critic.set_param('tree_method', 'hist')
            self.actor.set_param('gpu_id', -1)
            self.critic.set_param('gpu_id', -1)
            self.actor.set_param('predictor', 'cpu_predictor')
            self.critic.set_param('predictor', 'cpu_predictor')

    def __call__(self, obs, _=None):
        return self.forward(obs, _)

    def forward(self, obs, _=None):
        device = obs.device
        obs = obs.cpu().numpy()
        if self.actor is None:
            p = np.ones((obs.shape[0], self.action_length), dtype=np.float32) / self.action_length
            wp = np.ones(obs.shape[0], dtype=np.float32) / 2
        else:
            from xgboost import DMatrix
            xgb_obs = DMatrix(obs)
            p = self.actor.predict(xgb_obs)
            wp = self.critic.predict(xgb_obs)
        v = wp * 2 - 1
        pt = torch.log(torch.from_numpy(p).to(device))
        vt = torch.from_numpy(v).unsqueeze(-1).to(device)
        return {'policy': pt, 'value': vt}

    def load(self, path):
        import pickle
        with open(path + '-actor.xgb', 'rb') as f:
            self.actor = pickle.load(f)
        with open(path + '-critic.xgb', 'rb') as f:
            self.critic = pickle.load(f)

    def save(self, path):
        import pickle
        with open(path + '-actor.xgb', 'wb') as f:
            pickle.dump(self.actor, f)
        with open(path + '-critic.xgb', 'wb') as f:
            pickle.dump(self.critic, f)
