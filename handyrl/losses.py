# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# algorithms and losses

from collections import deque

import torch
import torch.nn.functional as F


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


def nash_vtrace(values, returns, rewards, actions, tmasks, log_prob, log_prob_buf, log_p, log_p_reg, lmb, gamma, rho, c, eta):
    values = rewards if rewards is not None else torch.zeros_like(log_prob)
    rewards = rewards if rewards is not None else torch.zeros_like(values)
    log_prob_ratio_buf = log_prob - log_prob_buf
    prob_ratio = torch.exp(log_prob_ratio_buf)
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1)

    v_hat = deque([returns[:, -1]])
    v_next = deque([returns[:, -1]])
    r_hat = deque([returns[:, -1] * 0])
    xi = deque([returns[:, -1] * 0])

    for i in range(values.size(1) - 1, -1, -1):
        prob_ratio_ = prob_ratio[:, i]
        value = values[:, i]
        rho_ = torch.clamp(prob_ratio_ * xi[0], 0, rho)
        c_ = torch.clamp(prob_ratio_ * xi[0], 0, c)

        delta_v = rho_ * (rewards[:, i] + prob_ratio_ * r_hat[0] + v_next[0] - value)

        tmask = tmasks[:, i]
        v_hat.appendleft(tmask * (values[:, i] + delta_v + c_ * (values_t_plus_1[:, i] - v_next[0])) + (1 - tmask) * v_hat[0])
        v_next.appendleft(tmask * values[:, i] + (1 - tmask) * v_next[0])
        r_hat.appendleft(tmask * 0 + (1 - tmask) * (rewards[:, i] + prob_ratio_ * r_hat[0]))
        xi.appendleft(tmask * 1 + (1 - tmask) * prob_ratio_ * xi[0])

    v_hat = torch.stack(tuple(v_hat), dim=1)
    r_hat = torch.stack(tuple(r_hat), dim=1)

    print(v_hat.shape)
    print('v =', v_hat[0])
    print('r =', r_hat[0])

    one_hot_actions = F.one_hot(actions, num_classes=log_p.size(-1)).squeeze(-2).to(actions.device)
    log_p_ratio_reg = log_p - log_p_reg
    log_prob_ratio_reg = log_p_ratio_reg.gather(-1, actions)
    log_prob_ratio_reg_opp = log_prob_ratio_reg.sum(-2, keepdim=True) - log_prob_ratio_reg
    prob_buf = torch.exp(log_prob_buf)

    modified_rewards = rewards + eta * log_prob_ratio_reg - eta * log_prob_ratio_reg_opp
    advantages = -eta * log_p_ratio_reg + one_hot_actions / prob_buf * (modified_rewards + prob_ratio * (v_hat[:, 1:] + r_hat[:, 1:]) - values)
    return v_hat[:, 1:], advantages


def nash_vtrace2(values, returns, rewards, actions, tmasks, log_prob, log_prob_buf, log_p, log_p_reg, lmb, gamma, rho, c, eta):
    values = rewards if rewards is not None else torch.zeros_like(log_prob)
    rewards = rewards if rewards is not None else torch.zeros_like(values)
    log_prob_ratio_buf = log_prob - log_prob_buf
    prob_ratio = torch.exp(log_prob_ratio_buf)
    values_t_plus_1 = torch.cat([values[:, 1:], returns[:, -1:]], dim=1)
    rhos = torch.clamp(prob_ratio, 0, 1e2)
    cs = torch.clamp(prob_ratio, 0, c)
    deltas = rhos * (rewards + gamma * values_t_plus_1 - values)

    # compute Vtrace value target recursively
    vs_minus_v_xs = deque([deltas[:, -1]])
    for i in range(values.size(1) - 2, -1, -1):
        vs_minus_v_xs.appendleft(deltas[:, i] + gamma * lmb * cs[:, i] * vs_minus_v_xs[0])

    vs_minus_v_xs = torch.stack(tuple(vs_minus_v_xs), dim=1)
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[:, 1:], returns[:, -1:]], dim=1)
    base_advantages = rewards + gamma * vs_t_plus_1 - values

    one_hot_actions = F.one_hot(actions, num_classes=log_p.size(-1)).squeeze(-2).to(actions.device)
    log_p_ratio_reg = log_p - log_p_reg
    log_prob_ratio_reg = log_p_ratio_reg.gather(-1, actions)
    log_prob_ratio_reg_opp = log_prob_ratio_reg.sum(-2, keepdim=True) - log_prob_ratio_reg
    prob_buf = torch.exp(log_prob_buf)

    modified_rewards = rewards# - eta * log_prob_ratio_reg_opp #+ eta * log_prob_ratio_reg
    advantages = -eta * log_p_ratio_reg + one_hot_actions * torch.clamp(1 / prob_buf, 1, 1e2) * (modified_rewards + rhos * base_advantages)
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
