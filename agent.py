import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections import deque
from VBLayer import VBLinear
from elastic_weight_consolidation import ElasticWeightConsolidation

# torch.backends.cudnn.enabled=False

device = torch.device("cuda")
import os


class Actor(nn.Module):
    def __init__(self, lr_model, use_cuda, frames, inputs, actions):
        super(Actor, self).__init__()

        self.input = inputs
        self.action = actions
        self.use_cuda = use_cuda
        self.frames = frames

        self.conv1 = nn.Conv2d(
            in_channels=self.frames, out_channels=64, kernel_size=8, stride=2
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1
        )
        self.bn4 = nn.BatchNorm2d(64 * 4)
        self.pooldown = nn.MaxPool2d(4, 4)
        self.firsl = VBLinear(4096, 1024)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256 * 4, kernel_size=3, stride=1
        )
        self.bn7 = nn.BatchNorm2d(256 * 4)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.lin1 = VBLinear(9216, 1024)
        self.bayes = VBLinear(2048, self.action)
        self.actor_eval_optim = torch.optim.Adam(self.parameters(), lr_model)
        self.to(torch.device("cuda") if self.use_cuda else torch.device("cpu"))

    def forward(self, states):
        with torch.autograd.set_detect_anomaly(True):
            x = self.pool1(self.relu(self.bn1(self.conv1(states))))
            x = self.bn4(self.conv4(x))
            x1 = self.firsl(torch.flatten(self.pooldown(x), 1))
            x = self.bn7(self.conv7(x))
            x = self.pool2(x)
            x = torch.flatten(x, 1)
            x = torch.sigmoid(self.lin1(x))
            x = torch.cat((x, x1), dim=1)
            x = torch.sigmoid(self.bayes(x))

            return x


class Critic(nn.Module):
    def __init__(self, lr_model, use_cuda, frames, inputs, actions, kl_weight):
        super(Critic, self).__init__()

        self.input = inputs
        self.action = actions
        self.use_cuda = use_cuda
        self.frames = frames
        self.kl_weight = kl_weight

        self.conv1 = nn.Conv2d(
            in_channels=self.frames, out_channels=64, kernel_size=8, stride=2
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64 * 4, kernel_size=3, stride=1
        )
        self.bn4 = nn.BatchNorm2d(64 * 4)
        self.pooldown = nn.MaxPool2d(4, 4)
        self.firsl = VBLinear(4096, 1024)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256 * 4, kernel_size=3, stride=1
        )
        self.bn7 = nn.BatchNorm2d(256 * 4)

        self.pool2 = nn.MaxPool2d(4, 4)

        self.lin1 = VBLinear(9216, 1024)
        self.alin1 = VBLinear(self.action, 1024)
        self.bayes = VBLinear(3072, self.action)
        self.critic_eval_optim = ElasticWeightConsolidation(
            self, nn.MSELoss(), self.kl_weight, lr_model
        )
        self.to(torch.device("cuda") if self.use_cuda else torch.device("cpu"))

    def forward(self, states, actions):
        with torch.autograd.set_detect_anomaly(True):
            x = self.pool1(self.relu(self.bn1(self.conv1(states))))
            x = self.bn4(self.conv4(x))
            x1 = self.firsl(torch.flatten(self.pooldown(x), 1))
            x = self.bn7(self.conv7(x))
            x = self.pool2(x)

            x = torch.flatten(x, 1)
            x = torch.sigmoid(self.lin1(x))
            x = torch.cat((x, x1), dim=1)
            y = self.alin1(actions)
            x = torch.cat((x, y), dim=1)
            x = torch.sigmoid(self.bayes(x))

            return x


class DDPG(object):
    def __init__(
        self,
        s_dim,
        a_dim,
        memo_capacity,
        lr_model,
        kl_weight,
        gamma,
        n_step,
        use_cuda,
        frames,
        update_params,
    ):

        self.use_cuda = use_cuda
        self.frames = frames
        self.first = True
        self.first2 = True
        self.update_params = update_params
        self.param1 = 0
        self.param2 = 0

        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.memory = deque(maxlen=memo_capacity)
        self.step_memory = deque(maxlen=memo_capacity)

        self.memo_capacity = memo_capacity
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.kl_weight = kl_weight
        self.gamma = gamma
        self.n_step = n_step
        self.lr_model = lr_model

        # actor nets
        self.actor_eval = self._build_actor().to(self.device)

        self.actor_eval2 = self._build_actor().to(self.device)

        self.critic_eval = self._build_critic().to(self.device)

        self.critic_eval2 = self._build_critic().to(self.device)

    def stack_frames(self, stacked_frames, state, is_new_episode, max_frames):

        # This check is applied if the episode is new, to maintain consistency, we still stack the inital frame
        if is_new_episode:
            stacked_frames = deque(
                [np.zeros((84, 84), dtype=np.int32) for i in range(self.frames)],
                maxlen=max_frames,
            )

            maxframe = np.maximum(state, state)
            stacked_frames.extend([maxframe] * max_frames)
            stacked_state = np.stack(stacked_frames, axis=1)

        # Stack the incomming frame with the three previous ones seen. 
        else:
            maxframe = np.maximum(stacked_frames[-1], state)
            stacked_frames.append(maxframe)
            stacked_state = np.stack(stacked_frames, axis=1)

        # Normalize the input between frames
        normalizer = nn.BatchNorm2d(self.frames)

        return normalizer(torch.FloatTensor(stacked_state)), stacked_frames

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)


        # Using double DDPG, we take the average action of the two agents
        action = self.actor_eval(state).cpu()
        action = action.data.numpy()

        action2 = self.actor_eval2(state).cpu()
        action2 = action2.data.numpy()

        return (action + action2) / 2

    def _build_actor(self):
        net = Actor(self.lr_model, self.use_cuda, self.frames, self.s_dim, self.a_dim)
        return net

    def _build_critic(self):
        net = Critic(
            (self.lr_model * 0.1),
            self.use_cuda,
            self.frames,
            self.s_dim,
            self.a_dim,
            self.kl_weight,
        )
        return net

    # We shift with a 50% chance between updating the first two networks or the two second
    def learn(self, batch_size):

        if np.random.uniform(0, 1) >= 0.5:
            loss = self.actor_learn(batch_size)
            self.critic_learn(batch_size)
            return loss, 1

        else:
            loss = self.actor_learn2(batch_size)
            self.critic_learn2(batch_size)
            return loss, 2

    def critic_learn(self, batch_size):

        sample_index = np.random.choice(len(self.memory), size=batch_size)

        b_memory = np.array([self.memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)

        # model TD target for backpropagation 
        Qsa = (
            self.actor_eval.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )

        next_actions = self.actor_eval2(batch_next_states)
        q_next = self.critic_eval2.forward(batch_next_states, next_actions)
        q_target = batch_rewards + (self.gamma * q_next)

        self.critic_eval.critic_eval_optim.forward_backward_update(
            (batch_states, Qsa), q_target
        )

        b_memory = np.array([self.step_memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)

        Qsa = (
            self.actor_eval.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )

        next_actions = self.actor_eval2(batch_next_states)
        q_next = self.critic_eval2.forward(batch_next_states, next_actions)
        q_target = batch_rewards + (self.gamma**self.n_step) * q_next

        # 
        if self.first is True or self.param1 % self.update_params == 0:

            self.first = False
            self.param1 += 1
            self.mem_list = []

            for _ in range(self.update_params):
                b_memory = np.array([self.step_memory[x] for x in sample_index])

                batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
                    self.device
                )

                batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(
                    self.device
                )

                batch_next_states = torch.FloatTensor(
                    np.array(b_memory[:, 3].tolist())
                ).to(self.device)

                batch_rewards = b_memory[:, 2].tolist()
                batch_rewards = (
                    torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)
                )

                batch_states = np.squeeze(batch_states, 1)
                batch_next_states = np.squeeze(batch_next_states, 1)

                Qsa = (
                    self.actor_eval.forward(batch_states)
                    .gather(1, batch_actions.long())
                    .to(self.device)
                )

                next_actions = self.actor_eval2(batch_next_states)
                q_next = self.critic_eval2.forward(batch_next_states, next_actions)

                q_target = batch_rewards + (self.gamma**self.n_step) * q_next
                self.mem_list.append(((batch_states, Qsa), q_target))

            self.critic_eval.critic_eval_optim.register_ewc_params(
                self.mem_list, batch_size, self.update_params
            )
        self.critic_eval.critic_eval_optim.forward_backward_update(
            (batch_states, Qsa), q_target
        )

    def critic_learn2(self, batch_size):
        sample_index = np.random.choice(len(self.memory), size=batch_size)

        b_memory = np.array([self.memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)
        Qsa = (
            self.actor_eval2.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )

        next_actions = self.actor_eval(batch_next_states)  # .detach()
        q_next = self.critic_eval.forward(batch_next_states, next_actions)  # .detach()
        q_target = batch_rewards + (self.gamma * q_next)

        self.critic_eval2.critic_eval_optim.forward_backward_update(
            (batch_states, Qsa), q_target
        )

        b_memory = np.array([self.step_memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)

        Qsa = (
            self.actor_eval2.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )

        next_actions = self.actor_eval(batch_next_states)
        q_next = self.critic_eval.forward(batch_next_states, next_actions)
        q_target = batch_rewards + (self.gamma**self.n_step) * q_next

        if self.first is True or self.param1 % self.update_params == 0:
            self.first = False
            self.param1 += 1
            self.mem_list = []
            for i in range(self.update_params):
                b_memory = np.array([self.step_memory[x] for x in sample_index])

                batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
                    self.device
                )

                batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(
                    self.device
                )

                batch_next_states = torch.FloatTensor(
                    np.array(b_memory[:, 3].tolist())
                ).to(self.device)

                batch_rewards = b_memory[:, 2].tolist()
                batch_rewards = (
                    torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)
                )

                batch_states = np.squeeze(batch_states, 1)
                batch_next_states = np.squeeze(batch_next_states, 1)

                Qsa = (
                    self.actor_eval2.forward(batch_states)
                    .gather(1, batch_actions.long())
                    .to(self.device)
                )

                next_actions = self.actor_eval(batch_next_states)  # .detach()
                q_next = self.critic_eval.forward(batch_next_states, next_actions)
                q_target = batch_rewards + (self.gamma**self.n_step) * q_next
                self.mem_list.append(((batch_states, Qsa), q_target))

            self.critic_eval2.critic_eval_optim.register_ewc_params(
                self.mem_list, batch_size, self.update_params
            )
        self.critic_eval2.critic_eval_optim.forward_backward_update(
            (batch_states, Qsa), q_target
        )

    def actor_learn(self, batch_size):
        sample_index = np.random.choice(len(self.memory), size=batch_size)

        b_memory = np.array([self.memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)
        Qsa = (
            self.actor_eval.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )
        q_expected = (
            self.critic_eval.forward(batch_states, Qsa)
            + self.critic_eval2.forward(batch_states, Qsa)
        ) / 2
        loss = torch.mean(-q_expected) + ((self.actor_eval.bayes.KL() + self.actor_eval.firsl.KL() + self.actor_eval.lin1.KL()) / 3) 

        b_memory = np.array([self.step_memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)
        Qsa = (
            self.actor_eval.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )
        q_expected = (
            self.critic_eval.forward(batch_states, Qsa)
            + self.critic_eval2.forward(batch_states, Qsa)
        ) / 2
        n_loss = torch.mean(-q_expected) +  ((self.actor_eval.bayes.KL() + self.actor_eval.firsl.KL() + self.actor_eval.lin1.KL()) / 3) 
        loss += n_loss

        with torch.autograd.set_detect_anomaly(True):

            self.actor_eval.actor_eval_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor_eval.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.actor_eval.parameters(), 50)
            self.actor_eval.actor_eval_optim.step()
        return loss

    def actor_learn2(self, batch_size):

        sample_index = np.random.choice(len(self.memory), size=batch_size)

        b_memory = np.array([self.memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)
        Qsa = (
            self.actor_eval2.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )
        q_expected = (
            self.critic_eval.forward(batch_states, Qsa)
            + self.critic_eval2.forward(batch_states, Qsa)
        ) / 2
        loss = torch.mean(-q_expected) + ((self.actor_eval2.bayes.KL() + self.actor_eval2.firsl.KL() + self.actor_eval2.lin1.KL()) / 3) 

        b_memory = np.array([self.step_memory[x] for x in sample_index])

        batch_states = torch.FloatTensor(np.array(b_memory[:, 0].tolist())).to(
            self.device
        )

        batch_actions = torch.FloatTensor(b_memory[:, 1].tolist()).to(self.device)

        batch_next_states = torch.FloatTensor(np.array(b_memory[:, 3].tolist())).to(
            self.device
        )

        batch_rewards = b_memory[:, 2].tolist()
        batch_rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)

        batch_states = np.squeeze(batch_states, 1)
        batch_next_states = np.squeeze(batch_next_states, 1)
        Qsa = (
            self.actor_eval2.forward(batch_states)
            .gather(1, batch_actions.long())
            .to(self.device)
        )
        q_expected = (
            self.critic_eval.forward(batch_states, Qsa)
            + self.critic_eval2.forward(batch_states, Qsa)
        ) / 2
        n_loss = torch.mean(-q_expected) + ((self.actor_eval2.bayes.KL() + self.actor_eval2.firsl.KL() + self.actor_eval2.lin1.KL()) / 3) 
        loss += n_loss
        with torch.autograd.set_detect_anomaly(True):

            self.actor_eval2.actor_eval_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor_eval.parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.actor_eval2.parameters(), 50)
            self.actor_eval2.actor_eval_optim.step()
        return loss

    # Calculation of the n-step reward and next-state, which is needed as G in the update function
    def n_step_calc(self, step_memory, gamma):
        trans = step_memory[-1]
        reward = trans[-3]
        next_state = trans[-2]

        for transition in reversed(list(step_memory)[:-1]):
            rew = transition[-3]
            ns = transition[-2]

            reward = rew + reward * gamma
            next_state = ns

        return reward, next_state

    def store_transition(self, state, action, reward, next_state, done):
        self.step_memory.append((state, action, reward, next_state, done))
        if len(self.step_memory) < self.n_step:
            return ()

        reward, next_state = self.n_step_calc(self.step_memory, self.gamma)

        self.memory.append((state, action, reward, next_state, done))

    def load_model(self, path):
        self.actor_eval.load_state_dict(torch.load(os.path.join(path, "actor.pth")))
