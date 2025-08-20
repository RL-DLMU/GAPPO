import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

from common.metrics import Metrics
from common.registry import Registry
from environment import TSCEnv
from trainer.base_trainer import BaseTrainer



import torch

@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.his_length = Registry.mapping['trainer_mapping']['setting'].param['his_length']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        self.group_num = Registry.mapping['model_mapping']['setting'].param['group_num']

        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'rewards_p', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'rewards_p', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''

        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        '''if 同构路口：
        建立某个智能体
            else:
            建立某个智能体
            '''
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))


        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    # @profile
    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        '''写入csv'''
        name = self.agents[0].model_dict['name']
        graph = Registry.mapping['world_mapping']['graph_setting'].graph
        grouping = graph["grouping"]
        total_decision_num = 0
        flush = 0

        reward_list = []
        reward_l_list = []
        loss_list = []
        real_average_travel_time_list = []



        b = [0] * len(self.world.intersections)
        self.ag = self.agents[0]

        print("grouping situation: ", grouping)

        for e in range(self.episodes):
            self.metric.clear()
            last_obs = self.env.reset()  
            his_action = deque(maxlen=self.his_length)
            last_actions = np.stack([[0]*self.ag.sub_agents])
            last_act_log_probs = np.stack([[-2]*self.ag.sub_agents])
            last_his_obs = []

            for a in self.agents:
                a.reset()
            self.ag.decision = 0
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss_q = []
            episode_loss_a = []
            episode_loss_vae = []
            buffer_cur = []
            i = 0
            rnn_state = None

            last_rnn_state = np.zeros((self.ag.sub_agents, self.ag.sub_agents * self.ag.ob_length))
            rewards_list = deque(maxlen=self.group_num * self.action_interval)
            rewards_p_list = deque(maxlen=self.group_num * self.action_interval)




            rnn_states = []
            last_obs = np.stack([ag.get_ob() for ag in self.agents])




            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents]) 
                    index = int(i / self.action_interval % self.group_num)
                    index_after = int((i + self.action_interval) / self.action_interval % self.group_num)
                    group_index = grouping[index] 
                    group_index_after = grouping[index_after]  
                    while len(his_action) < self.his_length:
                        his_action.appendleft(np.array([8] * self.ag.sub_agents))
                    actions = [[]]
                    act_log_probs = [[]]

                    for idx, ag in enumerate(self.agents):
                        action, act_log_prob, rnn_state, w = ag.get_action(last_obs[idx], rnn_state, his_action, test=False)

                        for ii,act in enumerate(action):
                            if ii in group_index:
                                actions[0].append(act)
                                act_log_probs[0].append(act_log_prob[ii])

                            else:
                                actions[0].append(last_actions[0][ii])
                                act_log_probs[0].append(last_act_log_probs[0][ii])

                    actions = np.stack(actions)  
                    act_log_probs = np.stack(act_log_probs)  
                    last_actions[0][group_index] = actions[0][group_index]
                    last_act_log_probs[0][group_index] = act_log_probs[0][group_index]
                    his_action.append(actions.flatten())
                    message = np.stack(rnn_state)


                    for _ in range(self.action_interval):
                        obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                        rewards_p_list.append(np.stack(rewards_p))
                    rewards = np.mean(rewards_list, axis=0) 
                    rewards_p = np.mean(rewards_p_list, axis=0)
                    self.metric.update(rewards, rewards_p)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])

                    for idx, ag in enumerate(self.agents):
                        if index == 0:
                            buffer_cur.append((last_obs[idx], last_phase[idx], actions[idx], rewards, obs[idx], act_log_probs[idx], list(his_action)))
                            if i > self.action_interval * self.group_num:
                                for agent_idx in group_index:
                                    buffer_cur[-2][3][agent_idx] = rewards[agent_idx]
                                    buffer_cur[-2][4][agent_idx] = obs[idx][agent_idx]
                        else:
                            for agent_idx in group_index:

                                buffer_cur[-1][0][agent_idx] = last_obs[idx][agent_idx]
                                buffer_cur[-1][1][agent_idx] = last_phase[idx][agent_idx]
                                buffer_cur[-1][2][agent_idx] = actions[idx][agent_idx]
                                
                                if i > self.action_interval * self.group_num:
                                    buffer_cur[-2][3][agent_idx] = rewards[agent_idx]
                                    buffer_cur[-2][4][agent_idx] = obs[idx][agent_idx]

                                buffer_cur[-1][5][agent_idx] = act_log_probs[idx][agent_idx]


                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                    total_decision_num += 1
                    last_obs = obs

                if all(dones):
                    break
            cur_loss_q, cur_loss_a = self.ag.train(buffer_cur)  # TODO: training

            episode_loss_q.append(np.stack([cur_loss_q]))
            episode_loss_a.append(np.stack([cur_loss_a]))
            if len(episode_loss_q) > 0:
                mean_loss_q = np.mean(np.array(episode_loss_q))
                mean_loss_a = np.mean(np.array(episode_loss_a))
            else:
                mean_loss_q = 0
                mean_loss_a = 0
            if len(episode_loss_vae) > 0:
                mean_loss_vae = np.mean(np.array(episode_loss_vae))
            else:
                mean_loss_vae = 0

            rewards_l, rewards_f = self.metric.rewards_each()

            self.logger.info("step:{}/{}, q_loss:{}, a_loss:{}, q_loss_vae:{}, rewards_p:{}, rewards:{}, rewards_l:{},queue:{}, delay:{}, travel_time:{}, throughput:{}".format(i, self.steps,\
                mean_loss_q, mean_loss_a, mean_loss_vae, self.metric.rewards_p(), self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.delay(), self.metric.real_average_travel_time(), int(self.metric.throughput())))

            for j in range(len(self.world.intersections)):
                b[j] = self.metric.lane_queue()[0][j]
            if e % 20 == 0:
                for j in range(len(self.world.intersections)):
                    print(j, b[j])
                print(sorted(b,reverse=True))
                print("========================")


            if self.test_when_train:
                real_average_travel_time = self.train_test(e)
                rewards_l, rewards_f = self.metric.rewards_each()
                reward_list.append(self.metric.rewards())
                reward_l_list.append(rewards_l)
                real_average_travel_time_list.append(real_average_travel_time)

        self.ag.save_model(e=self.episodes)




    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        obs = self.env.reset()  

        graph = Registry.mapping['world_mapping']['graph_setting'].graph
        grouping = graph["grouping"]


        his_action = deque(maxlen=self.his_length)
        last_actions = np.stack([[0]*self.ag.sub_agents])
        last_act_log_probs = np.stack([[-2]*self.ag.sub_agents])


        rnn_state = None


        rewards_list = deque(maxlen=self.group_num * self.action_interval)
        rewards_p_list = deque(maxlen=self.group_num * self.action_interval)


        for i in range(self.test_steps):

            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents])  
                index = int(i / self.action_interval % self.group_num)
                index_after = int((i + self.action_interval) / self.action_interval % self.group_num)
                group_index = grouping[index]  
                group_index_after = grouping[index_after]  

                while len(his_action) < self.his_length:
                    his_action.appendleft(np.array([8] * self.ag.sub_agents))
                actions = [[]]
                act_log_probs = [[]]


                rnn_states = []
                last_obs = np.stack([ag.get_ob() for ag in self.agents])
                for idx, ag in enumerate(self.agents):
                    action, act_log_prob, rnn_state, w = ag.get_action(last_obs[idx], rnn_state, his_action, test=True)

                    for ii,act in enumerate(action):
                        if ii in group_index:
                            actions[0].append(act)
                            act_log_probs[0].append(act_log_prob[ii])
                        else:
                            actions[0].append(last_actions[0][ii])
                            act_log_probs[0].append(last_act_log_probs[0][ii])

                actions = np.stack(actions)  
                act_log_probs = np.stack(act_log_probs)  

                last_actions[0][group_index] = actions[0][group_index]
                last_act_log_probs[0][group_index] = act_log_probs[0][group_index]

                his_action.append(actions.flatten())
                message = np.stack(rnn_state)



                for _ in range(self.action_interval):
                    obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    rewards_p_list.append(np.stack(rewards_p))
                rewards = np.mean(rewards_list, axis=0)  
                rewards_p = np.mean(rewards_p_list, axis=0)
                self.metric.update(rewards, rewards_p)
                cur_phase = np.stack([ag.get_phase() for ag in self.agents])




            if all(dones):
                break
        rewards_l, rewards_f = self.metric.rewards_each()
        mega_l = Registry.mapping['model_mapping']['setting'].param['mega_l']
        mega_f = Registry.mapping['model_mapping']['setting'].param['mega_f']

        self.logger.info("Test step:{}/{}, travel time :{}, rewards_p:{}, rewards:{}, rewards_l:{}, queue:{}, delay:{}, throughput:{}".format(\
            e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards_p(), self.metric.rewards(), rewards_l,\
            self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
        self.writeLog("TEST", e, self.metric.real_average_travel_time(),
             self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.queue_l(),self.metric.delay(), self.metric.throughput())

        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
            if self.save_replay:
                self.env.eng.set_save_replay(True)
                self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
            else:
                self.env.eng.set_save_replay(False)

        drop_load=False
        if not drop_load:
            [ag.load_model(self.episodes) for ag in self.agents]

        grouping = [[0, 2, 5], [1, 3, 4]]
        self.ag = self.agents[0]


        his_action = deque(maxlen=self.his_length)
        last_actions = np.stack([[0]*self.ag.sub_agents])
        last_act_log_probs = np.stack([[-2]*self.ag.sub_agents])
        rnn_state = None
        rewards_list = deque(maxlen=self.group_num * self.action_interval)
        rewards_p_list = deque(maxlen=self.group_num * self.action_interval)

        self.metric.clear()
        last_obs = self.env.reset()

        for i in range(self.test_steps):

            if i % self.action_interval == 0:
                last_phase = np.stack([ag.get_phase() for ag in self.agents])  
                index = int(i / self.action_interval % self.group_num)
                index_after = int((i + self.action_interval) / self.action_interval % self.group_num)
                group_index = grouping[index]  
                group_index_after = grouping[index_after]  

                while len(his_action) < self.his_length:
                    his_action.appendleft(np.array([8] * self.ag.sub_agents))
                actions = [[]]
                act_log_probs = [[]]


                rnn_states = []
                for idx, ag in enumerate(self.agents):
                    action, act_log_prob, rnn_state, w = ag.get_action(last_obs[idx], rnn_state, his_action, test=True)

                    for ii,act in enumerate(action):
                        if ii in group_index:
                            actions[0].append(act)
                            act_log_probs[0].append(act_log_prob[ii])
                        else:
                            actions[0].append(last_actions[0][ii])
                            act_log_probs[0].append(last_act_log_probs[0][ii])

                actions = np.stack(actions) 
                act_log_probs = np.stack(act_log_probs) 

                last_actions[0][group_index] = actions[0][group_index]
                last_act_log_probs[0][group_index] = act_log_probs[0][group_index]

                his_action.append(actions.flatten())
                message = np.stack(rnn_state)



                for _ in range(self.action_interval):
                    obs, rewards, rewards_p, dones, _ = self.env.step(actions.flatten())
                    i += 1
                    rewards_list.append(np.stack(rewards))
                    rewards_p_list.append(np.stack(rewards_p))
                rewards = np.mean(rewards_list, axis=0)  
                rewards_p = np.mean(rewards_p_list, axis=0)
                self.metric.update(rewards, rewards_p)
                cur_phase = np.stack([ag.get_phase() for ag in self.agents])

                last_obs = obs



                if all(dones):
                    break
                rewards_l, rewards_f = self.metric.rewards_each()


                self.logger.info(
                        "Test step:{}/{}, travel time :{}, rewards_p:{}, rewards:{}, rewards_l:{}, queue:{}, delay:{}, throughput:{}".format( \
                            int(i/10), self.episodes, self.metric.real_average_travel_time(), self.metric.rewards_p(),
                            self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.delay(),
                            int(self.metric.throughput())))
                self.writeLog("TEST", int(i/10), self.metric.real_average_travel_time(), \
                                  self.metric.rewards(), rewards_l, self.metric.queue(), self.metric.queue_l(), self.metric.delay(),
                                  self.metric.throughput())

        return self.metric

    def writeLog(self, mode, step, loss_q, loss_a, travel_time, cur_rwd, cur_queue, cur_delay, cur_throughput, none_='''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''):
        none_
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(step) + '\t' + "%.1f" % loss_q + '\t' + "%.1f" % loss_a + '\t' + "%.1f" % travel_time + '\t' +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()

