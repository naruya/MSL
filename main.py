import gym
import os
import torch
import argparse
import pickle
import gtimer as gt
import numpy as np
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append("./dssm")
from train import main as train_ssm
from diayn.examples.diayn import get_algorithm, experiment
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.util import DIAYNRollout as diayn_rollout
from rlkit.samplers.util import rollout as random_rollout
from rlkit.policies.simple import RandomPolicy


def collect(env, diayn, depth, args):
    if depth == 0:
        random_policy = RandomPolicy(env.action_space)
    else:
        diayn_policy = diayn.eval_data_collector.get_snapshot()['policy']

    data = []
    for skill in tqdm(range(args.skill_dim)):
        for trial in range(100):
            # print("skill-{} rollout-{}".format(skill, trial))
            if depth == 0:
                path = random_rollout(
                    env,
                    random_policy,
                    max_path_length=args.H,
                    render=False,
                )
            else:
                path = diayn_rollout(
                    env,
                    diayn_policy,
                    skill,
                    max_path_length=args.H,
                    render=False,
                )
            data.append([path['actions'], path['next_observations']])

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    train_path = os.path.join(args.data_dir, "./train{}.pkl".format(depth))
    test_path = os.path.join(args.data_dir, "./test{}.pkl".format(depth))

    os.makedirs(args.data_dir, exist_ok=True)
    with open(train_path, mode='wb') as f:
        pickle.dump(train_data, f)
    with open(test_path, mode='wb') as f:
        pickle.dump(test_data, f)


def update_sim(depth, args):
    ssm, ssm_log = train_ssm("--H {} --depth {} --epochs 100".format(args.H, depth))
    sim = SimNormalizedBoxEnv(gym.make(str(args.env)), ssm, depth, args)
    return sim, ssm_log


def update_policy(diayn, sim, diayn_path, args):
    experiment(diayn, sim, sim, args)
    file = os.path.join(diayn_path, "params.pkl")
    diayn, diayn_log = get_algorithm(env, env, args.skill_dim, file=file)
    return diayn, diayn_log


class SimNormalizedBoxEnv(NormalizedBoxEnv):
    def __init__(self, env, ssm, depth, args):
        super(SimNormalizedBoxEnv, self).__init__(env)
        self.ssm = ssm
        with open(os.path.join(args.data_dir, "param{}.pkl".format(depth)),
                  mode='rb') as f:
            self.a_mean, self.a_std, self.o_mean, self.o_std = \
                pickle.load(f)
        self.env.step = self.step
        self.envreset = self.env.reset
        self.env.reset = self.reset

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        # wrapped_step = self._wrapped_env.step(scaled_action)
        # next_obs, reward, done, info = wrapped_step
        # if self._should_normalize:
        #     next_obs = self._apply_normalize_obs(next_obs)

        a = scaled_action.astype(np.float32)
        a = (a - self.a_mean) / self.a_std
        a = torch.from_numpy(np.array([a]))
        o = self.ssm.step(a)[0]
        o = o.cpu().detach().numpy()
        next_obs = o * self.o_std + self.o_mean

        # return next_obs, reward * self._reward_scale, done, info
        return next_obs, 0, False, {}

    def reset(self, **kwargs):
        o_original = self.envreset(**kwargs)
        o = o_original.astype(np.float32)
        o = (o - self.o_mean) / self.o_std
        o = torch.from_numpy(np.array([o]))
        o = self.ssm.reset(o)
        return o_original
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str,
                        help='environment')
    parser.add_argument("--data_dir", type=str,
                        default="./data/")
    parser.add_argument('--skill_dim', type=int, default=100,
                        help='skill dimension')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--D', type=int, default=5,
                        help='Depth (The number of update)')
    args = parser.parse_args()

    env = NormalizedBoxEnv(gym.make(str(args.env)))
    diayn, diayn_log = get_algorithm(env, env, args.skill_dim)

    loghist = []
    for depth in range(args.D):
        collect(env, diayn, depth, args)
        sim, ssm_log = update_sim(depth, args)
        diayn, diayn_log = update_policy(diayn, sim, diayn_log, args)
        loghist.append([ssm_log, diayn_log])
        gt.reset()
    print(loghist)
    
    with open(datetime.now().strftime("%b%d_%H-%M-%S") + "_loghist.txt", mode='wb') as f:
        pickle.dump(loghist, f)