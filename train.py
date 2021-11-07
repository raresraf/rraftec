import argparse
import itertools
import pickle
import random
from collections import deque
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as O

from src.crafter_wrapper import Env


class DQN:
    def __init__(
            self,
            estimator,
            buffer,
            optimizer,
            epsilon_schedule,
            action_num,
            gamma=0.90,
            update_steps=1,
            update_target_steps=10,
            warmup_steps=10000,
    ):
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps = update_steps
        self._update_target_steps = update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert warmup_steps > self._buffer._batch_size, (
            "You should have at least a batch in the ER.")

    def act(self, state):
        with torch.no_grad():
            return self._estimator(state).argmax()

    def step(self, state):
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()

        if next(self._epsilon) < torch.rand(1).item():
            with torch.no_grad():
                qvals = self._estimator(state)
                return qvals.argmax()
        else:
            return torch.randint(self._action_num, (1,)).item()

    def learn(self, state, action, reward, state_, done):
        self._buffer.push((state, action, reward, state_, done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            batch = self._buffer.sample()
            self._update(*batch)

        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(self, states, actions, rewards, states_, done):
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class DoubleDQN(DQN):
    def _update(self, states, actions, rewards, states_, done):
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)

        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        obs = obs.reshape(1, obs.size(0), obs.size(1), obs.size(2))
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            obs = obs.reshape(1, obs.size(0), obs.size(1), obs.size(2))
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    net = get_estimator(env.action_space.n, opt.device)

    if opt.net == 'dqn':
        print("Using DQN net")
        agent = DQN(
            net,
            ReplayMemory(opt.device, size=1000, batch_size=32),
            O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=100000),
            env.action_space.n,
            warmup_steps=10000,
            update_steps=1,
        )
    elif opt.net == 'ddqn':
        print("Using Double DQN net")
        agent = DoubleDQN(
            net,
            ReplayMemory(opt.device, size=1000, batch_size=32),
            O.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=100000),
            env.action_space.n,
            warmup_steps=100,
            update_steps=1,
            update_target_steps=4
        )

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            state, done = env.reset().clone(), False
            state = state.reshape(1, state.size(0), state.size(1), state.size(2))

        action = agent.step(state)
        state_, reward, done, info = env.step(action)
        state_ = state_.reshape(1, state_.size(0), state_.size(1), state_.size(2))

        agent.learn(state, action, reward, state_, done)

        state = state_.clone()

        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--net",
        type=str,
        default='dqn',
        metavar="NET",
        help="Type of DQN",
    )
    return parser.parse_args()


class ReplayMemory:
    def __init__(self, device, size=1000, batch_size=32):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size
        self.device = device

    def push(self, transition):
        self._buffer.append(transition)

    def sample(self):
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        return (
            torch.cat(s, 0).to(self.device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.cat(s_, 0).to(self.device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self._buffer)


class ByteToFloat(nn.Module):
    def forward(self, x):
        assert (
                x.dtype == torch.uint8
        ), "The model expects states of type ByteTensor."
        return x.float().div_(255)


class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))


def get_estimator(action_num, device, input_ch=4, lin_size=32):
    return nn.Sequential(
        nn.Conv2d(input_ch, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        View(),
        nn.Linear(8 * 80 * 80, 8 * 80),
        nn.ReLU(inplace=True),
        nn.Linear(8 * 80, lin_size),
        nn.ReLU(inplace=True),
        nn.Linear(lin_size, action_num),
    ).to(device)


if __name__ == "__main__":
    main(get_options())
