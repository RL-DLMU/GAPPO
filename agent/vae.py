from . import RLAgent
from common.registry import Registry
import math
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
from collections import OrderedDict, deque
import gym
from torch.distributions import Categorical, Normal
from keras.utils import to_categorical
from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops

# device = torch.device("cuda", 1 if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
'''one-softmax；leader 的前驱是最后一个，最后一个是无效动作'''

@Registry.register_model('time_shift')
class CoLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching
        # self.buffer_size = Registry.mapping['trainer_mapping']['trainer_setting'].param['buffer_size']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.replay_buffer_vae = deque(maxlen=self.buffer_size)
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph

        self.world = world
        # self.sub_agents = 1
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for CoLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            # LaneVehicleGenerator通过这个生成的I
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get queue generator for CoLightAgent
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator for CoLightAgent
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))

        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases)
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['get_attention']
        # train parameters
        self.rank = rank
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.her = Registry.mapping['model_mapping']['setting'].param['her']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        # self.epsilon = np.ones(self.sub_agents) * Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        # self.lr_decay = Registry.mapping['model_mapping']['setting'].param['lr_decay']
        # self.lr_min = Registry.mapping['model_mapping']['setting'].param['lr_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        self.his_obs = deque(maxlen=self.his_length)
        self.his_action = deque(maxlen=self.his_length)
        while len(self.his_obs) < self.his_length:
            x_obs = []
            for i in range(self.sub_agents):
                x_obs.append([0.] * self.ob_length)
            self.his_obs.appendleft(np.array(x_obs))
        while len(self.his_action) < self.his_length:
            action = to_categorical(np.array(0), num_classes=self.action_space.n)
            self.his_action.appendleft(np.array([action] * self.sub_agents))
        # load_model = Registry.mapping['logger_mapping']['setting'].param['load_model']
        # if load_model:
        #     self.load_model(e=200)
        self.model = self._build_model().to(device)
        self.target_model = self._build_model().to(device)
        self.vae = self._build_vae().to(device)
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean').to(device)
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)
        # self.optimizer = optim.Adam(self.model.parameters(),
        #                                lr=self.learning_rate)

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for CoLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    '''观测中包含了邻居的历史动作'''
    # def get_his_obs(self, obs):
    #     self.his_obs.append(obs)
    #     Adj_matrix_noself = self.graph['Adj_matrix_noself']
    #     his_obs = []
    #     his_actions = []
    #     for item in self.his_action:
    #         expanded_item = np.repeat(item, self.sub_agents, axis=0)
    #         his_actions.append(expanded_item.reshape(self.sub_agents, -1, item.shape[-1]))
    #     his_aciton = []
    #     for j in range(self.his_length):
    #         his_aciton.append(Adj_matrix_noself @ his_actions[j])
    #
    #     for i in range(self.sub_agents):
    #         x_obs = []
    #         x_obs.append([lst[i] for lst in self.his_obs])
    #         x_obs = np.array(x_obs).reshape(1, -1)
    #         # x_obs[0].append(np.array([lst[i] for lst in his_aciton]))
    #         x_action = []
    #         x_action.append([lst[i] for lst in his_aciton])
    #         x_action = np.array(x_action).reshape(1, -1) # 邻居的历史动作onehot（his_length*4*action_dim）
    #         his_obs.append(np.concatenate((x_obs,x_action),axis=1)[0])
    #     return np.array(his_obs, dtype=np.float32) #前his_length项是自己的历史观测信息，最后一项是邻居的历史动作onehot（his_length,4,action_dim）

    def get_his_obs(self, obs):
        self.his_obs.append(obs)
        his_obs = []

        for i in range(self.sub_agents):
            x_obs = []
            x_obs.append([lst[i] for lst in self.his_obs])
            x_obs = np.array(x_obs[0], dtype=np.float32)
            # x_obs = np.array(x_obs).reshape(1, -1)
            # x_obs[0].append(np.array([lst[i] for lst in his_aciton]))
            his_obs.append(x_obs)
        return his_obs
        # return np.transpose(his_obs,(0,2,1))

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)
        # construct edge information.
        length = set([len(i) for i in x_obs])
        if len(length) == 1:  # each intersections may has  different lane nums
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x, axis=0) for x in x_obs]
        his_obs = self.get_his_obs(x_obs)

        return x_obs, his_obs

    def get_reward(self):
        # TODO: test output
        rewards_p = []
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):  # 得到每个路口的平均值 [16]
            rewards_p.append(self.reward_generator[i][1].generate())
            rewards.append(self.reward_generator[i][1].generate_reward())
        rewards_p = np.squeeze(np.array(rewards_p)) * 12  # *12,即乘以了每个路口的12条道 【16】里面的每一项是每个路口12条道的总和
        rewards = np.squeeze(np.array(rewards))
        return [rewards_p], [rewards_p]
        # return [rewards_p], [rewards_p]

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape) == 2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay  # [intersections,]

    def autoregreesive_act(self, observation, his_a, his_adj, model, batch_size, train):
        nodelayer_BFS = self.graph['nodelayer_BFS']
        edge_nei = torch.tensor(self.graph['Adj_matrix']).to(device)
        edge_pre = torch.tensor(self.graph['precursor_matrix']).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)
        node_BFS = torch.tensor(self.graph['node_BFS']).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)
        node_BFS_ = torch.zeros_like(node_BFS)
        for i in range(len(node_BFS)):
             node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)

        his_a.view(batch_size, -1, his_a.size(-1))
        shifted_action = torch.zeros((batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        output_q = torch.zeros((batch_size, self.sub_agents, self.action_space.n), dtype=torch.float32)
        output_action = torch.zeros((batch_size, self.sub_agents, 1), dtype=torch.long)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空

        '''处理his_a,转为one_hot,his_a不包括当前时刻动作，所以输入到网络里应该加上前驱当前时刻动作
            当前的his_a是所有路口的历史动作，加上前驱当前时刻的动作，作为网络的输入,不用每一轮挑出对应的前驱的历史动作，因为输入了前驱矩阵，在网络中再挑出来，且parallel_act无法每一轮挑'''
        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1)
        his_a = torch.cat((his_a, padd_agent), dim=-1) # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n+1)
        # one_hot_his_action = [[0] + sublist for sublist in one_hot_his_a]

        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(batch_size, -1, one_hot_his_a.shape[-2], one_hot_his_a.shape[-1])

        # his_aciton = torch.concatenate((his_a, shifted_action), dim=-2)
        for i, nodes in enumerate(nodelayer_BFS):
            out = model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix, node_BFS_, train)
            logit = out[:, nodes, :]
            output_q[:, nodes, :] = logit

            distri = Categorical(logits=logit)
            action = distri.probs.argmax(dim=-1)

            output_action[:, nodes, :] = action.unsqueeze(-1)
            if i + 1 < len(nodelayer_BFS):
                # shifted_action[:, nodes, 0] = 0  # 这个位置的动作不为空了
                one_hot_action = torch.tensor(F.one_hot(action, num_classes=self.action_space.n), dtype=torch.float32)
                shifted_action[:, nodes, :-1] = one_hot_action

                his_a[:, :, nodes, :] = torch.roll(his_a[:, :, nodes, :], -1, dims=1)
                his_a[:, -1, nodes, :-1] = one_hot_action
                his_a[:, -1, nodes, -1] = 0

        return out, output_action

    def parallel_act(self, observation, actions, his_a, his_adj, train):
        actions = actions.view(self.batch_size, self.sub_agents)
        edge_nei = torch.tensor(self.graph['Adj_matrix']).to(device)
        edge_pre = torch.tensor(self.graph['precursor_matrix']).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)
        node_BFS = torch.tensor(self.graph['node_BFS']).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)
        nodelayer_BFS = self.graph['nodelayer_BFS']
        leader = torch.tensor(nodelayer_BFS[0]).to(device)
        follow = torch.tensor(sorted(np.concatenate(nodelayer_BFS[1:])), dtype=torch.int64).to(device)

        node_BFS_ = torch.zeros_like(node_BFS)
        for i in range(len(node_BFS)):
            node_BFS_[node_BFS[i]] = i
        node_BFS_ = F.one_hot(node_BFS_, num_classes=self.sub_agents)


        one_hot_action = F.one_hot(actions.squeeze(-1), num_classes=self.action_space.n)  # (batch, n_agent, action_dim)
        shifted_action = torch.zeros((self.batch_size, self.sub_agents+1, self.action_space.n+1)).to(device)
        shifted_action[:, -1, -1] = 1  # leader的前驱智能体是-1；前驱智能体不存在，所以0位标为1，（0位是无效位，后八位是有效位）,0位标志，代表这个动作为空
        shifted_action[:, :-1, :-1] = one_hot_action

        # his_a[:, :, :, :] = torch.roll(his_a[:, :, :, :], -1, dims=1)
        # his_a[:, -1, :, :-1] = one_hot_action
        # his_a[:, -1, :, -1] = 0

        padd_agent = torch.tensor([8]*his_a.size(0)).reshape(his_a.size(0), 1)
        his_a = torch.cat((his_a, padd_agent), dim=-1)  # [batch*4,17]
        one_hot_his_a = F.one_hot(his_a, num_classes=self.action_space.n + 1)
        his_a = torch.tensor(one_hot_his_a, dtype=torch.float32).reshape(self.batch_size, -1, one_hot_his_a.shape[-2],
                                                                         one_hot_his_a.shape[-1])
        out = self.model(observation, his_a, edge_nei, edge_pre, nei_pos, his_pos_matrix,node_BFS_, train)
        return out

    def get_his_aciton(self,x_action):
        action = np.argmax(self.his_action[-1], axis=1)
        grouping = Registry.mapping['world_mapping']['graph_setting'].graph['grouping']
        group_index = self.decision % len(grouping)
        start = True if group_index == 0 else False
        for i in grouping[group_index]:
            action[i] = x_action[i]
        action_onehot = to_categorical(np.array(action), num_classes=self.action_space.n)

        if start:
            self.his_action.append(action_onehot)
        else:
            self.his_action[-1] = action_onehot
        # while len(self.his_action) < self.his_length:
        #     self.his_action.appendleft(np.array([0] * self.sub_agents))
        # for i in range(len(self.his_obs)):
        #     x_action.append(self.his_obs[i][0])
        # return np.array(x_action, dtype=np.float32)
        self.decision += 1
        return action

        # action = to_categorical(np.array(action), num_classes=self.action_space.n)
        # self.his_action.append(action)

    def get_action(self, ob, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                # return np.random.randint(0, self.action_space.n, self.sub_agents)
                return self.sample()

        observation = torch.tensor([ob], dtype=torch.float32).to(device)

        out = self.vae(observation.to(device), train=False)
        out = self.model(observation.to(device), train=False)
        # his_a = torch.tensor(his_a, dtype=torch.long).to(device)
        # his_adj = torch.tensor(his_adj, dtype=torch.long).to(device)
        # logit, output_action = self.autoregreesive_act(observation, his_a, his_adj, self.model, 1, train=False)

        out = out.squeeze().clone().cpu().detach().numpy()
        x_action = np.argmax(out, axis=1)
        # action = self.get_his_aciton(x_action)
        return x_action

    def sample(self):
        # action = np.random.randint(0, 1, self.sub_agents)
        x_action = np.random.randint(0, self.action_space.n, self.sub_agents)

        # grouping = Registry.mapping['world_mapping']['graph_setting'].graph['grouping'][0]
        # group_index = self.ag.decision % len(grouping)
        # start = True if group_index==0 else False
        # for i in grouping[group_index]:
        #     action[i] = x_action[i]
        # action = self.get_his_aciton(x_action)
        return x_action

    def _build_model(self):
        model = Decoder(self.ob_length, self.action_space.n, self.sub_agents, **self.model_dict)
        return model

    def _build_vae(self):
        vae = Variable_AutoEncoder(self.sub_agents)
        return vae

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))
    def remember_vae(self, last_obs, key):
        self.replay_buffer_vae.append((key, (last_obs)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        his_a = []
        his_adj = []
        rewards = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp[0], dtype=torch.float32)
            # a_pre = torch.tensor(dp[6], dtype=torch.float32)
            # a_pre = torch.tensor(np.tile(dp[6], (len(dp[0]), 1)), dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32)
            # a_pre = torch.tensor(dp[6], dtype=torch.float32)
            # a_pre_p = torch.tensor(np.tile(dp[7], (len(dp[4]), 1)), dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)  # (agents*batch_size,obs_dim+agent_dim)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        # try:
        #     a = torch.tensor(np.array(actions), dtype=torch.long)
        # except:
        #     actions = torch.tensor(np.array(actions), dtype=torch.long)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        # 虽然sub_agents==1,但是依然是集中训练，所以，将获得所有智能体的信息，信息拉平
        # if self.sub_agents > 1:
        #     rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        #     actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        actions = actions.view(actions.shape[0] * actions.shape[1])
        # his_adj = his_adj.view(his_adj.shape[0] * his_adj.shape[1],  his_adj.shape[2])
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions

    def _batchwise_vae(self, samples):
        # load onto tensor

        batch_list = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp, dtype=torch.float32)
            batch_list.append(Data(x=state))
        batch_t = Batch.from_data_list(batch_list)

        return batch_t


    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)
        # out, _ = self.autoregreesive_act(b_tp.x.to(device), his_a.to(device), his_adj.to(device), self.target_model, self.batch_size, train=False)
        out = self.target_model(b_tp.x.to(device), train=False)
        out = out.reshape(-1, self.action_space.n)
        target = rewards.to(device) + self.gamma * torch.max(out, dim=-1)[0]
        # target_f = self.parallel_act(b_t.x.to(device), actions.to(device), his_a.to(device), his_adj.to(device), train=False).reshape(-1, self.action_space.n)
        target_f = self.model(b_t.x.to(device), train=False).reshape(-1, self.action_space.n)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        # loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)

        deltas = target_f - self.model(b_t.x.to(device), train=True).reshape(-1, self.action_space.n)
        # deltas = target_f - self.model(x=b_t.x.to(device), edge_index=edge, train=True)

        real_deltas = torch.where(deltas > 0, deltas, deltas * self.her)
        loss = torch.mean(torch.pow(real_deltas.to(device), 2)).to(device)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().cpu().detach().numpy()

    def train_vae(self):
        samples = random.sample(self.replay_buffer_vae, self.batch_size)
        x = self._batchwise_vae(samples)
        x = x.x
        x_hat, m = self.vae(x.to(device))  # 模型的输出，在这里会自动调用model中的forward函数
        loss = self.loss_function(x_hat, x)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        return loss.clone().cpu().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len, d_model=128, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)  ## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
#         pe[:, 1::2] = torch.cos(position * div_term)  ##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
#         ## 上面代码获取之后得到的pe:[max_len*d_model]
#
#         ## 下面这个代码之后，我们得到的pe形状是：[max_len*1*d_model]
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以
#
#     def forward(self, x):
#         """
#         x: [seq_len, batch_size, d_model]
#         """
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        # TimeBlock输入为[50, 207, 12, 2]即[B,H,W,C]
        # 转换后的结果为[50, 2, 207, 12]，即[B,C,H,W]；与卷积核对应，因为卷积核大小为（1，kernel_size=3，分别对应H,W)，C表示通道，B表示batch的大小
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X)) #TimeBlock计算的结果是[50, 64, 207, 10]，即[B,C,H,W]（以调用一次TimeBlock为例，W减少2）
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1) #再进行一次维度变换，返回的结果是[50, 207, 10, 64],即[B,H,W,C],因为后面还会调用TimeBlock，所以输入格式保持一致
        return out

class TimeBlock_decoder(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock_decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.ConvTranspose2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        # TimeBlock输入为[50, 207, 12, 2]即[B,H,W,C]
        # 转换后的结果为[50, 2, 207, 12]，即[B,C,H,W]；与卷积核对应，因为卷积核大小为（1，kernel_size=3，分别对应H,W)，C表示通道，B表示batch的大小
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X)) #TimeBlock计算的结果是[50, 64, 207, 10]，即[B,C,H,W]（以调用一次TimeBlock为例，W减少2）
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1) #再进行一次维度变换，返回的结果是[50, 207, 10, 64],即[B,H,W,C],因为后面还会调用TimeBlock，所以输入格式保持一致
        return out
class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X) # t:[50, 207, 10, 64],即[B,H,W,C]
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1)) # t2:[50, 207, 10, 16],即[B,H,W,C]
        t3 = self.temporal2(t2) # t3:[50, 207, 8, 64]
        return self.batch_norm(t3) # 输出结果为[50, 207, 8, 64]，经过BN其输出维度保持不变
        # return t3
class STGCNBlock_decoder(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock_decoder, self).__init__()
        self.temporal1 = TimeBlock_decoder(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock_decoder(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X) # t:[50, 207, 10, 64],即[B,H,W,C]
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1)) # t2:[50, 207, 10, 16],即[B,H,W,C]
        t3 = self.temporal2(t2) # t3:[50, 207, 8, 64]
        return self.batch_norm(t3) # 输出结果为[50, 207, 8, 64]，经过BN其输出维度保持不变
'''输入[B,H,W,C]；B：batch H:网格节点数量，W表示特征数量(时间片数量)，C：特征数（只有队列长度就12,12条lane）'''
class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    # num_features = 1（就队列长度）
    # num_timesteps_input：20（100秒，每五秒存一次，那么就是20次）num_timesteps_output：5（前100秒预测后25秒）
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,  #减去2 * 5因为过了总共过了五个时间层（block中各俩，TimeBlock中一个），每过一个时间层，W就少2
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat) # 输出结果为[50, 207, 8, 64]
        out2 = self.block2(out1, A_hat) # 输出结果为[50, 207, 4, 64]
        out3 = self.last_temporal(out2) # 输出结果为[50, 207, 2, 64]
        # out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))) # 输出结果为[50, 207, 3]
        # return out4
        return out3.reshape((out3.shape[0], out3.shape[1], -1))

'''输入[B,H,W,C]；B：batch H:网格节点数量，W表示特征数量(时间片数量)，C：特征数（只有队列长度就12,12条lane）'''
class STGCN_decoder(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """
    # num_features = 1（就队列长度）
    # num_timesteps_input：20（100秒，每五秒存一次，那么就是20次）num_timesteps_output：5（前100秒预测后25秒）
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN_decoder, self).__init__()
        self.block1 = STGCNBlock_decoder(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock_decoder(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock_decoder(in_channels=64, out_channels=num_features)
        self.fully = nn.Linear(num_timesteps_input, (num_timesteps_output - 2 * 5) * 64)  #减去2 * 5因为过了总共过了五个时间层（block中各俩，TimeBlock中一个），每过一个时间层，W就少2

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.fully(X)  # 输出结果为[50, 207, 3]
        out1 = out1.reshape(out1.shape[0], out1.shape[1], -1, 64)
        out2 = self.block1(out1, A_hat) # 输出结果为[50, 207, 8, 64]
        out3 = self.block2(out2, A_hat) # 输出结果为[50, 207, 4, 64]
        out4 = self.last_temporal(out3) # 输出结果为[50, 207, 2, 64]

        return out4
        # return out3.reshape((out3.shape[0], out3.shape[1], -1))
# def gconv(x, theta, Ks, c_in, c_out):
#     '''
#     Spectral-based graph convolution function.
#     :param x: tensor, [batch_size, n_route, c_in].
#     :param theta: tensor, [Ks*c_in, c_out], trainable kernel parameters.
#     :param Ks: int, kernel size of graph convolution.
#     :param c_in: int, size of input channel.
#     :param c_out: int, size of output channel.
#     :return: tensor, [batch_size, n_route, c_out].
#     '''
#     # graph kernel: tensor, [n_route, Ks*n_route]
#     kernel = tf.get_collection('graph_kernel')[0]
#     n = tf.shape(kernel)[0]
#     # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
#     x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
#     # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
#     x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
#     # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
#     x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
#     # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
#     x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
#     return x_gconv
#
# def layer_norm(x, scope):
#     '''
#     Layer normalization function.
#     :param x: tensor, [batch_size, time_step, n_route, channel].
#     :param scope: str, variable scope.
#     :return: tensor, [batch_size, time_step, n_route, channel].
#     '''
#     _, _, N, C = x.get_shape().as_list()
#     mu, sigma = tf.nn.moments(x, axes=[2, 3], keep_dims=True)
#
#     with tf.variable_scope(scope):
#         gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
#         beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
#         _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
#     return _x
#
# # 时间卷积层
# def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu'):
#     '''
#     Temporal convolution layer.
#     :param x: tensor, [batch_size, time_step, n_route, c_in].
#     :param Kt: int, kernel size of temporal convolution.
#     :param c_in: int, size of input channel.
#     :param c_out: int, size of output channel.
#     :param act_func: str, activation function.
#     :return: tensor, [batch_size, time_step-Kt+1, n_route, c_out].
#     '''
#     _, T, n, _ = x.get_shape().as_list()
#
#     if c_in > c_out:
#         w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
#         tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
#         x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
#     elif c_in < c_out:
#         # if the size of input channel is less than the output,
#         # padding x to the same size of output channel.
#         # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
#         x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
#     else:
#         x_input = x
#
#     # keep the original input for residual connection.
#     x_input = x_input[:, Kt - 1:T, :, :]
#
#     if act_func == 'GLU':
#         # gated liner unit
#         wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
#         tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
#         bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
#         x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
#         return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
#     else:
#         wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
#         tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
#         bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
#         x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
#         if act_func == 'linear':
#             return x_conv
#         elif act_func == 'sigmoid':
#             return tf.nn.sigmoid(x_conv)
#         elif act_func == 'relu':
#             return tf.nn.relu(x_conv + x_input)
#         else:
#             raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')
#
# # 空间卷积层
# def spatio_conv_layer(x, Ks, c_in, c_out):
#     '''
#     Spatial graph convolution layer.
#     :param x: tensor, [batch_size, time_step, n_route, c_in].
#     :param Ks: int, kernel size of spatial convolution.
#     :param c_in: int, size of input channel.
#     :param c_out: int, size of output channel.
#     :return: tensor, [batch_size, time_step, n_route, c_out].
#     '''
#     _, T, n, _ = x.get_shape().as_list()
#
#     if c_in > c_out:
#         # bottleneck down-sampling
#         w_input = tf.get_variable('ws_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
#         tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w_input))
#         x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
#     elif c_in < c_out:
#         # if the size of input channel is less than the output,
#         # padding x to the same size of output channel.
#         # Note, _.get_shape() cannot convert a partially known TensorShape to a Tensor.
#         x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
#     else:
#         x_input = x
#
#     ws = tf.get_variable(name='ws', shape=[Ks * c_in, c_out], dtype=tf.float32)
#     tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
#     variable_summaries(ws, 'theta')
#     bs = tf.get_variable(name='bs', initializer=tf.zeros([c_out]), dtype=tf.float32)
#     # x -> [batch_size*time_step, n_route, c_in] -> [batch_size*time_step, n_route, c_out]
#     x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
#     # x_g -> [batch_size, time_step, n_route, c_out]
#     x_gc = tf.reshape(x_gconv, [-1, T, n, c_out])
#     return tf.nn.relu(x_gc[:, :, :, 0:c_out] + x_input)
#
# # 时空卷积层
# def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU'):
#     '''
#     Spatio-temporal convolutional block, which contains two temporal gated convolution layers
#     and one spatial graph convolution layer in the middle.
#     :param x: tensor, batch_size, time_step, n_route, c_in].
#     :param Ks: int, kernel size of spatial convolution.
#     :param Kt: int, kernel size of temporal convolution.
#     :param channels: list, channel configs of a single st_conv block.
#     :param scope: str, variable scope.
#     :param keep_prob: placeholder, prob of dropout.
#     :param act_func: str, activation function.
#     :return: tensor, [batch_size, time_step, n_route, c_out].
#     '''
#     c_si, c_t, c_oo = channels
#
#     with tf.variable_scope(f'stn_block_{scope}_in'):
#         x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func)
#         x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)
#     with tf.variable_scope(f'stn_block_{scope}_out'):
#         x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo)
#     x_ln = layer_norm(x_o, f'layer_norm_{scope}')
#     return tf.nn.dropout(x_ln, keep_prob)

# 定义变分自编码器VAE
class Variable_AutoEncoder(nn.Module):

    def __init__(self,n_node):

        super(Variable_AutoEncoder, self).__init__()
        # self.STGCN = STGCN(n_node, 12, 20, 5)
        self.num_timesteps_input = 20 #每100秒训练一次，每5秒控制一次，所以一共20个历史步
        self.num_timesteps_output = 5
        self.n_node = n_node
        # 定义编码器
        # self.Encoder = nn.Sequential(
        #     nn.Linear(784, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64),
        #     nn.ReLU()
        # )
        self.Encoder = STGCN(n_node, 12, self.num_timesteps_input, self.num_timesteps_output)
        self.Decoder = STGCN_decoder(n_node, 12, self.num_timesteps_output, self.num_timesteps_input)
        # 定义解码器
        # self.Decoder = nn.Sequential(
        #     nn.Linear(20, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 784),
        #     nn.Sigmoid()
        # )

        self.fc_m = nn.Linear((self.num_timesteps_input - 2 * 5) * 64,  #减去2 * 5因为过了总共过了五个时间层（block中各俩，TimeBlock中一个），每过一个时间层，W就少2
                               self.num_timesteps_output)
        self.fc_sigma = nn.Linear((self.num_timesteps_input - 2 * 5) * 64,  #减去2 * 5因为过了总共过了五个时间层（block中各俩，TimeBlock中一个），每过一个时间层，W就少2
                               self.num_timesteps_output)

    def _forward(self, input_x):
        Adj_matrix_noself = Registry.mapping['world_mapping']['graph_setting'].graph['Adj_matrix_noself']
        Adj_matrix = torch.tensor(Adj_matrix_noself, dtype=torch.float32).to(device)
        # x = self.STGCN(Adj_matrix, input_x)

        input_x = input_x.view(-1, self.n_node, input_x.size(-2),input_x.size(-1))
        code = self.Encoder(Adj_matrix, input_x)

        # m, sigma = code.chunk(2, dim=1)
        m = self.fc_m(code) # 原有编码
        sigma = self.fc_sigma(code) # 控制噪音干扰程度的编码，为随机噪音码(e1,e2,e3)分配权重

        e = torch.randn_like(sigma) # e为随机噪音码

        c = torch.exp(sigma) * e + m # exp(σi)的目的是为了保证这个分配的权重是个正值 将原编码与噪音编码相加，就得到了VAE在code层的输出结果
        # c = sigma * e + m

        output = self.Decoder(Adj_matrix, c)
        output = output.reshape(-1, output.size(-2), output.size(-1))

        return output, m

    def forward(self, obs, train=True):
        if train:
            return self._forward(obs)
        else:
            with torch.no_grad():
                return self._forward(obs)


def softmax_mod(input, dim=None):
    if dim is None:
        dim = input.dim() - 1
        # 将输入张量减去每行的最大值，以防止指数溢出
    input_max, _ = torch.max(input, dim=dim, keepdim=True)
    input_exp = torch.exp(input)
    # 计算指定维度上的softmax分母，即每行元素的指数值之和
    denominator = torch.sum(input_exp, dim=dim, keepdim=True)
    # 执行softmax计算，除以分母得到最终结果
    output = input_exp / (denominator + 1)
    return output

def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2], X.shape[3])
    X = X.permute(0, 2, 3, 1, 4)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 3, 1, 2, 4)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])

class MultiHeadAttModel(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding_nei = positional_encoding(6, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        self.positional_encoding_pre = positional_encoding(5, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q ,k):
        # query = q[:, :, -1, :]
        query = q
        key = k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = q.size(-1)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        # neighbor_repr = torch.unsqueeze(key, dim=1)
        # neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1)  # [?, 16, 16, 32]
        # # shape2 = neighbor_repr.shape
        # # print(shape2)
        # # if adjs != None:
        # # adjs = adjs.repeat(batch_size, 1, 1)
        # # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]
        #
        # if adjs_pos != None:
        #
        #     flag = 'nei' if adjs_pos.size(-1) == 6 else 'pre'
        #     adjs_pos = adjs_pos.repeat(batch_size, 1, 1)
        #     adjs_pos = torch.reshape(adjs_pos, (-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1))) #[256,16,32,1,6]
        #     # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
        #     positional_encoding = self.positional_encoding_nei if flag == 'nei' else self.positional_encoding_pre
        #     pos_encoding = positional_encoding.repeat(adjs_pos.size(0), adjs_pos.size(1), adjs_pos.size(2), 1, 1)
        #     pos_encoding = (torch.matmul(adjs_pos, pos_encoding)).squeeze()
        #     # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
        #     neighbor_repr = neighbor_repr + pos_encoding
        # shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr = key
        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        # mask = adjs
        # mask = F.pad(mask, (1, 0))
        # mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 1)
        # att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

class MultiHeadAttModel_allin(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel_allin, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding_nei = positional_encoding(6, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        self.positional_encoding_pre = positional_encoding(16, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q, k, adjs_pre, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        neighbor_repr = key.reshape(key.size(0), -1, key.size(-1)) # (batch_size, 4[his]*17[n_agent+1], 128)
        neighbor_repr = torch.unsqueeze(neighbor_repr, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        # print(shape2)
        # if adjs != None:
        # adjs = adjs.repeat(batch_size, 1, 1)
        # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]

        if adjs_pos != None:

            flag = 'nei' if adjs_pos.size(-1) == 6 else 'pre'
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1)
            adjs_pos = torch.reshape(adjs_pos, (-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1))) #[256,16,32,1,6]
            # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
            positional_encoding = self.positional_encoding_nei if flag == 'nei' else self.positional_encoding_pre
            pos_encoding = positional_encoding.repeat(adjs_pos.size(0), adjs_pos.size(1), adjs_pos.size(2), 1, 1)
            pos_encoding = (torch.matmul(adjs_pos, pos_encoding)).squeeze()
            # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
            neighbor_repr = neighbor_repr + pos_encoding
        shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs_pre
        # mask = F.pad(mask, (1, 0))
        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 4)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat
class MultiHeadAttModel_his(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel_his, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding = positional_encoding(4, self.n_embd)  # 第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding = PositionalEncoding(4, self.n_embd)

    def forward(self, q, k, adjs):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)
        adjs = adjs.reshape(-1, 4)

        # create a 0 vector with same size as one quer vector in Q
        # zero_vector = torch.zeros(batch_size, 1, d_k).to(q.device)
        # query = torch.cat((zero_vector, q), dim=1)
        # key = torch.cat((zero_vector, k), dim=1)

        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape
        # print(shape1)

        # hj*Ws
        neighbor_repr = key.permute(0, 2, 1, 3)
        positional_encoding = self.positional_encoding.repeat(batch_size, n_agent, 1, 1)
        neighbor_repr = neighbor_repr + positional_encoding
        # neighbor_repr = torch.unsqueeze(key, dim=1)
        # neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1, 1)  # [?, 16, 16, 32]
        # shape2 = neighbor_repr.shape
        # print(shape2)
        # if adjs != None:
        # adjs = adjs.repeat(batch_size, 1, 1)
        # adjs = torch.reshape(adjs, (-1, n_agent, adjs.size(1), n_agent))
        # neighbor_repr = torch.matmul(adjs, neighbor_repr)  # [?, 16, 5, 32]

            # adjs_pos = torch.reshape(adjs_pos,(-1, n_agent, adjs_pos.size(-2), 1, adjs_pos.size(-1)))  # [256,16,32,1,6]
            # pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
        # pos_encoding = self.positional_encoding.repeat(n_agent, 1, 1)
            # pos_encoding = F.pad(pos_encoding, (0, 0, 1, 0))
        # neighbor_repr = neighbor_repr + pos_encoding
        shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        # neighbor_repr_head = transpose_qkv(neighbor_repr_head.reshape(neighbor_repr_head.shape[0],neighbor_repr_head.shape[1],-1,neighbor_repr_head.shape[-1]), self.nv)
        # neighbor_repr_head = neighbor_repr_head.reshape(neighbor_repr_head.shape[0],neighbor_repr_head.shape[1], 4, -1,neighbor_repr_head.shape[-1])
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]
        # print(neighbor_repr_head.shape)

        # att = torch.exp(torch.matmul(agent_repr_head, neighbor_repr_head))
        # att = att / (1 + torch.sum(att))
        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs
        # mask = F.pad(mask, (1, 0))
        mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(int(att.size(0)/mask.size(0)), att.size(1), 1, 1, 1)
        mask[:, -1, :, :, :] = torch.tensor([0, 0, 0, 1])
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)  # [?, 16, 1, 1, 5]

        # print("att:",att.shape)
        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]
        # print(neighbor_hidden_repr_head.shape)

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

# def add_pos(neighbor_repr):
#     pos_encoding = positional_encoding(neighbor_repr.size(-2), neighbor_repr.size(-1))
#     pos_encoding = pos_encoding.repeat(neighbor_repr.size(0), neighbor_repr.size(1), 1, 1)
#     pos = (torch.ones((pos_encoding.shape[0], pos_encoding.shape[1], pos_encoding.shape[2], pos_encoding.shape[3])) * 0)
#     pos_fill = (torch.ones((pos_encoding.shape[0], pos_encoding.shape[1], pos_encoding.shape[-1])) * 0)
#     graph = Registry.mapping['world_mapping']['graph_setting'].graph
#     node_precursor = graph['node_precursor']
#     for i, nodes in enumerate(node_precursor):
#         j = 0
#         a = [0, 1, 2, 3]
#         for node in nodes:
#             if node == -1:
#                 continue
#             elif node == i-1:
#                 pos[:, i, j, :] = pos_encoding[:, i, 0, :]
#                 a.remove(0)
#                 j += 1
#             elif node == i+1:
#                 pos[:, i, j, :] = pos_encoding[:, i, 1, :]
#                 a.remove(1)
#                 j += 1
#             elif node < i:
#                 pos[:, i, j, :] = pos_encoding[:, i, 2, :]
#                 a.remove(2)
#                 j += 1
#             else:
#                 pos[:, i, j, :] = pos_encoding[:, i, 3, :]
#                 a.remove(3)
#                 j += 1
#         for f in a:
#             pos[:, i, j, :] = pos_encoding[:, i, f, :]
#             j += 1
#         # while j < 4:
#         #     pos[:, i, j, :] = pos_fill
#         #     j += 1
#     neighbor_repr = neighbor_repr + pos
#
#     return neighbor_repr

# def positional_encoding(max_seq_len, d_model):
#     # 初始化位置编码矩阵
#     position = torch.arange(0, max_seq_len).unsqueeze(1).float()
#     div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#     pe = torch.zeros(max_seq_len, d_model)
#     # 使用正弦和余弦函数计算位置编码
#     pe[:, 0::2] = torch.sin(position * div_term)
#     pe[:, 1::2] = torch.cos(position * div_term)
#
#     padd = torch.zeros(d_model)
#     # 增加一个维度，以便将位置编码添加到词嵌入中
#     # pe = pe.unsqueeze(0)
#     return pe
class DecodeBlock_obs(nn.Module):

    def __init__(self, n_embd, dv, n_head, n_agent):
        super(DecodeBlock_obs, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU()
            # init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        q = x[:, :, -1, :]
        k = x
        x = q + self.attn(q, k)
        # x = self.mlp(x)
        return x


class DecodeBlock(nn.Module):

    def __init__(self, n_embd, dv, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = MultiHeadAttModel_allin(n_embd, dv, n_head, n_agent)
        self.attn2 = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.attn = MultiHeadAttModel(n_embd, dv, n_head, n_agent)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.Softmax(-1),
        )

    def forward(self, rep_enc, his_action_embeddings):
        # x = self.ln1(x + self.attn1(x, x, adjs_mates))
        x = self.attn(q=rep_enc, k=his_action_embeddings)

        # x = self.attn2(q=rep_enc, k=his_action_embeddings, adjs=adjs_pre, adjs_pos=adjs_pos)
        # x = self.mlp(x)
        x = rep_enc + x
        # x = rep_enc + self.attn2(q=rep_enc, k=x, adjs=adjs_pre)
        # x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):

    # def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, action_type='Discrete', dec_actor=False, share_actor=False):
    def __init__(self, input_dim, output_dim, n_node, **kwargs):
        super(Decoder, self).__init__()
        self.model_dict = kwargs
        self.action_dim = gym.spaces.Discrete(output_dim).n

        self.n_embd = self.model_dict.get('NODE_EMB_DIM')[0]
        self.obs_dim = input_dim
        self.n_agent = n_node
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        n_block = 1
        self.STGCN = STGCN(n_node, 12, 20, 5)
        self.action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim, self.n_embd, bias=False), activate=True), nn.GELU())
        self.his_action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim+1, self.n_embd, bias=False), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(init_(nn.Linear(self.obs_dim, self.n_embd), activate=True), nn.GELU())
        self.positional_encoding_obs = positional_encoding(self.his_length, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        # self.positional_encoding_action = positional_encoding(self.his_length, self.n_embd) #第一项为需要mask的向量，其他为上下左右自己向量
        self.ln = nn.LayerNorm(self.n_embd)
        self.blocks_obs = nn.Sequential(*[DecodeBlock_obs(self.model_dict.get('INPUT_DIM_1')[i],
                                                          self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                                          self.model_dict.get('NUM_HEADS')[i],
                                                          self.n_agent) for i in range(n_block)])
        self.blocks = nn.Sequential(*[DecodeBlock(self.model_dict.get('INPUT_DIM_1')[i],
                                                  self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                                  self.model_dict.get('NUM_HEADS')[i],
                                                  self.n_agent) for i in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.head_l = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.head_f = nn.Sequential(init_(nn.Linear(self.n_embd, self.action_dim), activate=True))
        self.val_head = nn.Sequential(init_(nn.Linear(self.n_embd, self.n_embd), activate=True), nn.GELU(),
                                      nn.LayerNorm(self.n_embd),
                                      init_(nn.Linear(self.n_embd, 1)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def _forward(self, input_x):
        Adj_matrix_noself = Registry.mapping['world_mapping']['graph_setting'].graph['Adj_matrix_noself']
        Adj_matrix = torch.tensor(Adj_matrix_noself, dtype=torch.float32).to(device)
        x = self.STGCN(Adj_matrix, input_x)

        input_x = torch.reshape(input_x, (-1, self.n_agent, input_x.shape[-1]))
        obs = input_x[:,:,:self.his_length*self.obs_dim]
        obs = obs.reshape(obs.shape[0], obs.shape[1], self.his_length, -1)
        nei_action = input_x[:,:,self.his_length*self.obs_dim:].reshape(-1,self.n_agent,self.his_length, 4, self.action_dim)
        nei_action = torch.sum(nei_action, dim=-2)

        obs_emb = self.obs_encoder(obs)
        positional_encoding = self.positional_encoding_obs.repeat(obs_emb.size(0),obs_emb.size(1), 1, 1).to(device)
        obs_emb_pos = obs_emb + positional_encoding
        for block in self.blocks_obs:
            obs_embeddings = block.forward(obs_emb_pos)

        nei_action_emb = self.action_encoder(nei_action)
        act_emb_pos = nei_action_emb + positional_encoding

        for block in self.blocks:
            x = block(obs_embeddings, act_emb_pos)
        out = self.head(x)
        return out

    def forward(self, obs, train=True):
        if train:
            return self._forward(obs)
        else:
            with torch.no_grad():
                return self._forward(obs)


def positional_encoding(max_seq_len, d_model):
    # 初始化位置编码矩阵
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    # 使用正弦和余弦函数计算位置编码
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # padd = torch.zeros(d_model).unsqueeze(0)
    # pe = torch.cat((padd, pe), dim=0)
    # 增加一个维度，以便将位置编码添加到词嵌入中
    # pe = pe.unsqueeze(0)
    return pe


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


