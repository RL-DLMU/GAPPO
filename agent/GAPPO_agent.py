from . import RLAgent
from common.registry import Registry
import numpy as np
import math
import os
import random
from collections import OrderedDict, deque
import gym

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
import itertools

device = torch.device("cpu")
# device = torch.device("cuda", 2 if torch.cuda.is_available() else "cpu")

@Registry.register_model('IPPO')
class IPPOAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching

        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph

        print('IPPO')
        # print(Registry.mapping['world_mapping']['graph_setting'])
        self.world = world
        self.sub_agents = len(self.world.intersections)  # 3
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long).to(device)  
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id] 
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  
        self.ob_generator = observation_generators


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
        print("==============================")
        print(self.ob_generator[0][1])
        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases) 
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length  # 12

        print("self.ob_length")
        print(self.ob_length)

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['get_attention']

        self.rank = rank  # 0
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.critic_lr = Registry.mapping['model_mapping']['setting'].param['critic_lr']
        self.actor_lr = Registry.mapping['model_mapping']['setting'].param['actor_lr']
        self.epoch = Registry.mapping['trainer_mapping']['setting'].param['epochs']
        self.eps = Registry.mapping['model_mapping']['setting'].param['eps']
        self.lmbda = Registry.mapping['model_mapping']['setting'].param['lambda']

        self.message_gamma = Registry.mapping['model_mapping']['setting'].param['message_gamma']

        self.model = self._build_model()
        self.target_model = self._build_model()

        self.actor = self._build_actor()

        self.criterion = nn.MSELoss(reduction='mean')
        self.critic_optimizer = torch.optim.Adam(itertools.chain(self.model.parameters()), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.Adam(itertools.chain(self.actor.parameters()), lr=self.actor_lr)


    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]  
            node_idx = self.graph['node_id2idx'][node_id] 
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0]) 
        self.ob_generator = observation_generators

        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0]) 
        self.reward_generator = rewarding_generators


        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0]) 
        self.phase_generator = phasing_generators


        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))

        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def get_ob(self):
        x_obs = []  
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)

        length = set([len(i) for i in x_obs])
        if len(length) == 1: 
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x,axis=0) for x in x_obs]
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards = []  
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards)) * 12
        return rewards, rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))  
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase  # [0 0 0]

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay))
        return delay 

    def get_action(self, ob, rnn_state, his_action, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        observation = torch.tensor(ob, dtype=torch.float32).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)  

        his_action = torch.tensor(his_action, dtype=torch.long).to(device)
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)

        if rnn_state is not None:
            rnn_state = torch.tensor(rnn_state, dtype=torch.float32).to(device)
        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)

        action_dist, rnn_state = self.actor(observation, rnn_state, his_action, nei_pos, test)
        if test==False:
            action = action_dist.sample()
            action_log_probs = action_dist.log_prob(action).to(device)
        if test==True:
            action = torch.argmax(action_dist.probs, dim=1)
            action_log_probs = action_dist.log_prob(action).to(device)

        return action.view(-1).cpu().clone().numpy(), action_log_probs.view(-1).cpu().clone().detach().numpy(), rnn_state.cpu().clone().detach().numpy(), None


    def _build_model(self):
        model = ColightNet(self.ob_length, self.action_space.n, **self.model_dict).to(device)
        return model
    
    def _build_actor(self):
        model = Actor(self.ob_length, self.action_space.n, self.his_length, self.message_gamma, **self.model_dict).to(device)
        return model
    
    def remember(self, last_obs, last_phase, actions, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))


    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        his_action = []
        rewards = []
        act_log_prob = torch.empty((0), dtype=torch.double).to(device)
        for item in samples:
            dp = item
            act_log_prob = torch.cat((act_log_prob, torch.tensor(dp[5]).to(device)), 0)
            state = torch.tensor(dp[0], dtype=torch.float32).to(device) 
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32).to(device) 
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
            his_action.append(dp[6])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(device)
        his_action = torch.tensor(np.array(his_action), dtype=torch.long).to(device)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
            his_action = his_action.view(his_action.shape[0], his_action.shape[1], his_action.shape[2])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions, act_log_prob, his_action

    def train(self, transition_buffer):

        action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases)).n
        his_pos_matrix = torch.tensor(self.graph['his_pos_matrix']).to(device)
        nei_pos = torch.tensor(self.graph['nei_pos_matrix']).to(device)
        max_episode_len = len(transition_buffer)
        b_t, b_tp, rewards, actions, act_log_prob, his_action = self._batchwise(transition_buffer)
        his_action = torch.tensor(his_action).to(device)

        obs = b_t.x
        obs = obs.view(max_episode_len, self.sub_agents, -1)
        obs_next = b_tp.x
        out_next = self.model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        target = rewards.view(-1, 1) + self.gamma * out_next.view(-1, 1)

        out = self.model(x=b_t.x, edge_index=b_tp.edge_index, train=True)

        td_error = (target - out).view(max_episode_len, self.sub_agents, -1)
        td_error = td_error.permute(1, 0, 2)
        adv = torch.empty((0), dtype=torch.float32).to(device)
        for item in td_error:  
            advantage = compute_advantage(self.gamma, self.lmbda, item.cpu()).to(device) 
            adv = torch.cat((adv, advantage), 0)
        adv = adv.view(self.sub_agents, max_episode_len, 1)
        advantage = adv.permute(1, 0, 2)
        advantage = advantage.contiguous().view(-1, 1)
        actions = actions.view(max_episode_len, self.sub_agents, 1)
        actions = actions.type(torch.long)

        epochs = 15
        for _ in range(epochs):
            new_probs = torch.empty((0), dtype=torch.float32).to(device)
            rnn_state = None
            for i, item in enumerate(obs):
                inp = item.to(device)
                new_act, rnn_state = self.actor(inp, rnn_state, his_action[i], nei_pos, test=False)
                new_probs = torch.cat((new_probs, new_act.probs), 0)

            new_act_p = new_probs.view(max_episode_len, self.sub_agents, action_space)
            new_log_probs = torch.log(torch.gather(new_act_p, dim=2, index=actions)).to(device)
            ratio = torch.exp(new_log_probs.view(-1, 1) - act_log_prob.detach().view(-1, 1))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage

            log_new_act = torch.log(new_act_p)
            policy_entropy = - torch.sum(new_act_p * log_new_act, dim=2)
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - 0.01 * torch.mean(policy_entropy)
            new_out = self.model(x=b_t.x, edge_index=b_tp.edge_index, train=True)
            critic_loss = torch.mean(F.mse_loss(new_out, target.detach()))

            self.critic_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            self.critic_optimizer.step()
            self.actor_optimizer.step()

        torch.cuda.empty_cache()  
        return critic_loss.cpu().clone().detach().numpy(), actor_loss.cpu().clone().detach().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')

        self.actor = self._build_actor()
        self.actor.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.actor.state_dict(), model_name)


class ColightNet(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(ColightNet, self).__init__()
        self.model_dict = kwargs
        self.action_space = gym.spaces.Discrete(output_dim)  # 8
        self.features = input_dim  # 12
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, self.model_dict.get('RNN_DIM')[0])
        for i in range(self.model_dict.get('N_LAYERS')):  # 0
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i)
            self.module_list.append(block)
        output_dict = OrderedDict()

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                output_dict.update({name: h})
                name = f'relu_{l_idx}'
                output_dict.update({name: nn.ReLU})
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)  # 128, 8
        else:
            out = nn.Linear(block.d_out, 1)  # in:128, out:8
        name = f'output'
        output_dict.update({name: out})  # 一个线性层128，8
        self.output_layer = nn.Sequential(output_dict)
        self.dense_1 = nn.Linear(self.model_dict.get('RNN_DIM')[0], 128)

    def _forward(self, x, edge_index):
        h = self.embedding_MLP(x)
        h =F.relu(self.dense_1(h))
        h = self.output_layer(h)
        return h

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Embedding_MLP(nn.Module):
    def __init__(self, in_size, emb):  # layers: [128, 128]
        super(Embedding_MLP, self).__init__()
        self.embedding_node = nn.Sequential(
            layer_init(nn.Linear(in_size, emb)),
            nn.ReLU(),
            layer_init(nn.Linear(emb, emb)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.embedding_node(x)
        return x


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d  # 128
        self.dv = dv  # 16
        self.d_out = d_out  # 128
        self.nv = nv  # 5
        self.suffix = suffix  # 0
        self.W_target = nn.Linear(d, dv * nv)  # 128, 80
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)  # 16, 128
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index=edge_index)
        x1 = torch.unsqueeze(x, dim=0)
        x2 = x1.repeat(2, 1, 1)
        edge1 = F.one_hot(edge_index)
        re = torch.bmm(edge1.float(), x2)
        aggregated = self.message1(re[0], re[1], edge_index=edge_index) 
        out = self.out(aggregated)
        out = F.relu(out) 
        self.att = self.att_list
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message1(self, x_i, x_j, edge_index):  
        h_target = F.relu(self.W_target(x_i))  
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)  
        agent_repr = h_target.permute(1, 0, 2)  

        h_source = F.relu(self.W_source(x_j))  
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv) 
        neighbor_repr = h_source.permute(1, 0, 2) 

        index = edge_index[1] 

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  
        max_node = torch_scatter.scatter_max(e_i, index=index)[0] 
        max_i = max_node.index_select(1, index=index) 
        ec_i = torch.add(e_i, -max_i) 
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index) 
        normst_node = torch.add(norm_node, 1e-12)
        normst_i = normst_node.index_select(1, index)  

        alpha_i = ecexp_i / normst_i 
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0)) 

        hidden_neighbor = F.relu(self.hidden_embedding(x_j)) 
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv) 
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2) 
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0) 
        out_ = torch_scatter.scatter_sum(out, index=index, dim=0)

        return out_

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def positional_encoding(max_seq_len, d_model):
    position = torch.arange(0, max_seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe = torch.zeros(max_seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], num_heads, -1)
    X = X.permute(0, 3, 1, 2, 4)
    return X.reshape(-1, X.shape[2], X.shape[3], X.shape[4])

def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2], X.shape[3])
    X = X.permute(0, 2, 3, 1, 4)
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

class MultiHeadAttModel_h(nn.Module):
    def __init__(self, n_embd, dv, nv, n_agent):
        super(MultiHeadAttModel_h, self).__init__()
        self.n_embd = n_embd
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.n_agent = n_agent
        self.positional_encoding_nei = torch.tensor([3, 5, 7, 11, 13]) 
        self.positional_encoding_pre = positional_encoding(5, self.n_embd) 

    def forward(self, q, k, adjs):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        d_k = k.size(-1)


        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2) 
        shape1 = agent_repr_head.shape

        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1) 

        indices = torch.arange(neighbor_repr.size(1)).unsqueeze(1)
        neighbor_repr[:, indices, indices, :] = query[:, indices]
        shape3 = neighbor_repr.shape
        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3) 

        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        mask = adjs

        mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(att.size(0), 1, 1, 1, 1)
        att = att.masked_fill(mask == 0, -1e9)
        att = F.softmax(att, dim=-1)

        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2) 


        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2) 

        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2) 
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

class MultiHeadAttModel_allin(nn.Module):
    def __init__(self, n_embd, dv, nv, his_length):
        super(MultiHeadAttModel_allin, self).__init__()
        self.n_embd = n_embd
        self.action_dim = 8
        self.dv = dv
        self.nv = nv
        self.fcv = init_(nn.Linear(n_embd, dv * nv))
        self.fck = init_(nn.Linear(n_embd, dv * nv))
        self.fcq = init_(nn.Linear(n_embd, dv * nv))
        self.fcout = init_(nn.Linear(dv * nv, n_embd))
        self.his_action_encoder = nn.Sequential(init_(nn.Linear(self.action_dim + 1, self.n_embd, bias=False), activate=True), nn.GELU())

        self.positional_encoding = positional_encoding(his_length, self.n_embd) 

    def forward(self, q, k, adjs_pos=None):
        query, key = q, k
        n_agent = q.size(1)
        batch_size = q.size(0)
        # Get the dimension of key vectors (needed for scaling the dot product of Q and k)
        d_k = k.size(-1)


        # hi*Wt
        agent_repr = torch.unsqueeze(query, dim=2)
        agent_repr_head = F.relu(self.fcq(agent_repr))
        agent_repr_head = transpose_qkv(agent_repr_head, self.nv)
        agent_repr_head = torch.unsqueeze(agent_repr_head, dim=2)  # [?, 16, 1, 1, 32]
        shape1 = agent_repr_head.shape

        neighbor_repr = torch.unsqueeze(key, dim=1)
        neighbor_repr = neighbor_repr.repeat(1, n_agent, 1, 1, 1)  # [?, 16, 16, 32]


        if adjs_pos != None:
            adjs_pos = adjs_pos.reshape(adjs_pos.shape[0], -1, n_agent).unsqueeze(0)
            adjs_pos = adjs_pos.repeat(batch_size, 1, 1, 1).unsqueeze(-2).type(torch.float32)

            neighbor_repr = torch.matmul(adjs_pos, neighbor_repr)  # mean filed
            neighbor_repr = self.his_action_encoder(neighbor_repr.squeeze(-2))
            positional_encoding = self.positional_encoding.unsqueeze(0).unsqueeze(1).repeat(neighbor_repr.size(0),neighbor_repr.size(1),1,1).to(device)  # TP
            neighbor_repr = neighbor_repr + positional_encoding
        shape3 = neighbor_repr.shape
        # print(shape3)

        neighbor_repr_head = F.relu(self.fck(neighbor_repr))
        neighbor_repr_head = transpose_qkv(neighbor_repr_head, self.nv)
        neighbor_repr_head = torch.unsqueeze(neighbor_repr_head, dim=2).permute(0, 1, 2, 4, 3)  # [?, 16, 1, 5, 32]

        att = torch.matmul(agent_repr_head, neighbor_repr_head) / math.sqrt(d_k)
        att = F.softmax(att, dim=-1)

        att_record = torch.squeeze(att, dim=2)

        neighbor_hidden_repr_head = F.relu(self.fcv(neighbor_repr))
        neighbor_hidden_repr_head = transpose_qkv(neighbor_hidden_repr_head, self.nv)
        neighbor_hidden_repr_head = torch.unsqueeze(neighbor_hidden_repr_head, dim=2)  # [?, 16, 1, 5, 32]

        out = torch.mean(torch.matmul(att, neighbor_hidden_repr_head), dim=2)  # [?, 16, 1, 32]
        # print(out.shape)
        out_concat = transpose_output(out, self.nv)
        out_concat = torch.squeeze(out_concat, dim=2)  # [?, 16, 32]
        out_concat = F.relu(self.fcout(out_concat))

        return out_concat

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, his_length, mess_gamma,**kwargs): 
        super(Actor, self).__init__()
        self.model_dict = kwargs
        self.embedding_MLP = Embedding_MLP(input_dim, self.model_dict.get('RNN_DIM')[0])
        self.rnn_hidden = None
        self.rnn = nn.GRUCell(self.model_dict.get('RNN_DIM')[0], self.model_dict.get('RNN_DIM')[1])
        self.attn = MultiHeadAttModel_h(self.model_dict.get('RNN_DIM')[0],
                                            self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[0],
                                            self.model_dict.get('NUM_HEADS')[0],
                                            his_length)

        self.dense_1 = nn.Linear(self.model_dict.get('RNN_DIM')[0], output_dim)

        self.mess_gamma = mess_gamma


    def _forward(self, x, rnn_state, his_action, nei_pos):

        x = self.embedding_MLP(x)
        h = self.rnn(x, rnn_state)
        if rnn_state is not None:
            rep = h.view(-1, nei_pos.shape[0], rnn_state.shape[-1])
            message = rnn_state.view(-1, rnn_state.shape[0], rnn_state.shape[1])

            message_declined = self.mess_gamma * message

            x = self.attn(q=rep, k=message_declined, adjs=nei_pos)
            x = rep + x
            x = x.view(-1, rnn_state.shape[-1])
        else:
            x = h

        x = F.relu(self.dense_1(x))
        x = torch.distributions.Categorical(logits=x)

        return x, h


    def forward(self, x, rnn_state, his_action, nei_pos, test=False):
        if test:
            with torch.no_grad():
                return self._forward(x, rnn_state, his_action, nei_pos)
        else:
            return self._forward(x, rnn_state, his_action, nei_pos)



def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)