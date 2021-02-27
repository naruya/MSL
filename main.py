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


def collect(diayn, depth, args):
    if depth in [0, 1]:  # use old data (hard coding)
        return None

    if depth == 0:
        random_policy = RandomPolicy(env.action_space)
    else:
        diayn_policy = diayn.eval_data_collector.get_snapshot()['policy']

    env = NormalizedBoxEnv(gym.make(str(args.env)))

    data = []
    for skill in tqdm(range(args.skill_dim)):
        for trial in range(100):
            # print("skill-{} rollout-{}".format(skill, trial))
            if depth == 0:6
                path = random_rollout(
                    env,
                    random_policy,
                    max_path_length=args.H_collect,
                    render=False,
                )
            else:
                path = diayn_rollout(
                    env,
                    diayn_policy,
                    skill,
                    max_path_length=args.H_collect,
                    render=False,
                )
            data.append([path['actions'], path['next_observations']])

    train_data = data[:int(len(data)*0.9)]
    test_data = data[int(len(data)*0.9):]

    print(len(train_data), len(test_data))

    if not depth == 0:
        pre_train_path = os.path.join(args.data_dir, "loop{}/train.pkl".format(depth-1))
        pre_test_path = os.path.join(args.data_dir, "loop{}/test.pkl".format(depth-1))

        with open(pre_train_path, mode='rb') as f:
            pre_train_data = pickle.load(f)
        with open(pre_test_path, mode='rb') as f:
            pre_test_data = pickle.load(f)

        train_data = pre_train_data + train_data
        test_data = pre_test_data + test_data

    print(len(train_data), len(test_data))

    os.makedirs(os.path.join(args.data_dir, "loop{}/".format(depth)), exist_ok=True)

    train_path = os.path.join(args.data_dir, "loop{}/train.pkl".format(depth))
    test_path = os.path.join(args.data_dir, "loop{}/test.pkl".format(depth))

    with open(train_path, mode='wb') as f:
        pickle.dump(train_data, f)
    with open(test_path, mode='wb') as f:
        pickle.dump(test_data, f)


def update_sim(depth, args):
    if depth == 0:
        ssm, ssm_log = train_ssm(
            "--data ./data/loop{} --epochs {} --T 10 --B 256 \
             --timestamp Feb26_08-06-04 --load_epoch 2000".format(depth, args.ssm_epochs))
    elif depth == 1:
        ssm, ssm_log = train_ssm(
            "--data ./data/loop{} --epochs {} --T 10 --B 256 \
             --timestamp Feb27_09-34-00 --load_epoch 2000".format(depth, args.ssm_epochs))
    else:
        ssm, ssm_log = train_ssm(
            "--data ./data/loop{} --epochs {} --T 10 --B 256".format(depth, args.ssm_epochs))

    sim = SimNormalizedBoxEnv(gym.make(str(args.env)), ssm, depth, args)
    return sim, ssm_log


def update_diayn(sim, depth, args, diayn_log=None):
    if diayn_log:
        file = os.path.join(diayn_log, "params.pkl")
    else:
        file = None

    if depth == 0:  # hard coding!!
        diayn_log = '/root/workspace/MSL/diayn/data/DIAYN-100-HalfCheetah-v2/DIAYN_100_HalfCheetah-v2_2021_02_27_07_40_50_0000--s-0'
        file = os.path.join(diayn_log, "params.pkl")
        diayn = get_algorithm(sim,
                      sim,
                      args.skill_dim,
                      epochs=args.diayn_epochs,
                      length=args.H_diayn,
                      file=file)
        diayn.log_dir = diayn_log
        return diayn, diayn_log

    elif depth == 1:  # hard coding!!
        diayn_log = '/root/workspace/MSL/diayn/data/DIAYN-100-HalfCheetah-v2/DIAYN_100_HalfCheetah-v2_2021_02_27_11_01_03_0000--s-0'
        file = os.path.join(diayn_log, "params.pkl")
        diayn = get_algorithm(sim,
                      sim,
                      args.skill_dim,
                      epochs=args.diayn_epochs,
                      length=args.H_diayn,
                      file=file)
        diayn.log_dir = diayn_log
        return diayn, diayn_log

    # USE SIM !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    diayn = get_algorithm(sim,
                          sim,
                          args.skill_dim,
                          epochs=args.diayn_epochs,
                          length=args.H_diayn,
                          file=file)
    diayn_log = diayn.log_dir
    experiment(diayn)
    return diayn, diayn_log


class SimNormalizedBoxEnv(NormalizedBoxEnv):
    def __init__(self, env, ssm, depth, args):
        super(SimNormalizedBoxEnv, self).__init__(env)
        self.ssm = ssm
        path = os.path.join(args.data_dir, "loop{}/param.pkl".format(depth))
        with open(path, mode='rb') as f:
            self.a_mean, self.a_std, self.o_mean, self.o_std = \
                pickle.load(f)
        self.env.step = self.step
        self.envreset = self.env.reset
        self.env.reset = self.reset
        self.t = 0

    def step(self, action):
        self.t += 1
        if self.t % 100 == 0:
            print("sim", end="")  # now using simulator!

        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        a = scaled_action.astype(np.float32)
        a = (a - self.a_mean) / self.a_std
        a = torch.from_numpy(np.array([a]))
        o = self.ssm.step(a)[0]
        o = o.cpu().detach().numpy()
        next_obs = o * self.o_std + self.o_mean
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
    parser.add_argument('--H_collect', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--H_diayn', type=int, default=50,
                        help='Max length of DIAYN')
    parser.add_argument('--D', type=int, default=5,
                        help='Depth (The number of update)')
    parser.add_argument('--diayn_epochs', type=int, default=50)
    parser.add_argument('--ssm_epochs', type=int, default=2000)
    args = parser.parse_args()

    log_path = datetime.now().strftime("%b%d_%H-%M-%S") + ".log.txt"
    diayn, diayn_log, sim, sim_log = None, None, None, None

    for depth in range(args.D):
        collect(diayn, depth, args)
        del sim
        sim, sim_log = update_sim(depth, args)
        del diayn
        diayn, diayn_log = update_diayn(sim, depth, args, diayn_log)
        with open(log_path, mode='a') as f:
            f.write(str([sim_log, diayn_log]))
        gt.reset()