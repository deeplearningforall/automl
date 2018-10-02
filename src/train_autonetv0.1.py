import gym
from average_functions import DNN
import random
import numpy as np
from collections import deque
from datetime import datetime
from collections import defaultdict
from Environment import Environment
from AgentUtility import AgentUtility
import sys

"""
Lunar Lander Agent implementation using DDQN

usage: python3 train_lunarlander.py <learning_rate> <epsilon_step_size> <replay_buffer_size> <episodes> <optional: training_save_file_name>

The neural network is implemented in average_function.py
"""
class LunarLander:
    def __init__(self, gamma=0.995, learning_rate=0.001, epsilon_step_size=125, replay_buffer_size=50000):
        self.env = Environment()
        self.model = DNN(8,128,2,self.env.action_space.n, learning_rate)
        self.epsilon = 1.0
        self.min_exploration = 3000
        self.replay_buffer_size = replay_buffer_size
        self.exploration_set = deque(maxlen=(self.replay_buffer_size))
        self.exploration_ct = 0
        self.gamma = gamma
        self.replay_start_marker = 0
        self.transer_weight_count = 5
        self.replay_time = 1
        self.moving_avg = deque(maxlen=100)
        self.learning_rate = learning_rate
        self.epsilon_step_size = epsilon_step_size
        self.epsilon_stat = defaultdict(int)
        print ("Number of action space is: {0}, learning_rate is: {1}, gamma is {2}, epsilon_step_size is {3} and replay_buffer_size is {4}".format(self.env.action_space.n, self.learning_rate, self.gamma, self.epsilon_step_size, self.replay_buffer_size))

    """
    Method to render the environment
    """
    def render(self):
        pass

    """
    Method to train the agent. The number of epsiodes to run is controlled by episodes argument.

    Logic:
        At the begining of the episode, the environment is first reet. For each iteration, the getAction method is called
        to get the action. getAction method returns the action as per epsilon - greedy policy.
        Each time the agent takes an action, it receives reward and the trainsition state information. The infomration
        is stored in the replay buffer in a form of tuple - (x,action,reward,x_new,done).

        The agent calls replay_batch at the end of an episode, provided the experience collected in the replay buffer
        is greater than the minimum exploration count as configured. The default is 3000. The batch size is provided as 1024.

        Epsilon has starting value of 1.0. The epsilon is reduced by 0.05 every "step_size" number of episodes.
        Once the value reaches 0.1. It remains so until 3000th episode. At 3001st episode, epsilon is set to constant value of 0.05

        The agent keeps track of best score above 199 and is saved to save_path. The file that is available at the end of the training
        will be from the best moving average score above 199 as seen by the agent.
    """
    def act(self, episodes=1, render=False, save_path=None):
        outfile = open('training_reward_avg_per_episode_{0}_{1}_{2}_{3}.txt'.format(self.learning_rate, self.gamma, self.epsilon_step_size, self.replay_buffer_size),'w')
        logfile = open('training_{0}_{1}_{2}_{3}.log'.format(self.learning_rate, self.gamma, self.epsilon_step_size, self.replay_buffer_size),'w')
        best_score = 199
        for ep in range(episodes):
            agent_utility = AgentUtility()
            self.env.set_action_utility(agent_utility)
            index = 0
            x = self.env.reset_stats()
            rv = 0
            while True:
                index += 1
                if render:
                    self.render()
                action = self.getAction(x)
                x_new, reward, done = self.env.step(action)
                rv += reward
                self.exploration_set.append((x,action,reward,x_new,done))
                x = x_new
                self.exploration_ct += 1
                if done:
                    break
            self.moving_avg.append(rv)
            test_rw = np.average(self.moving_avg)
            #logfile.write('{4}:  Ran episode {0}, scored {1}, epsilon is {2}, average score {3}\n'.format(ep, rv, self.epsilon, test_rw, str(datetime.now())))
            print('{4}:  Ran episode {0}, scored {1}, epsilon is {2}, average score {3}'.format(ep, rv, self.epsilon, test_rw, str(datetime.now())))
            outfile.write("{0},{1},{2}\n".format(ep,np.average(self.moving_avg),rv))
            if self.exploration_ct > self.min_exploration and ep % self.replay_time == 0:
                self.replay_batch(ep, 1024)

            #self.epsilon_stat[self.epsilon] += 1
            if ep % self.epsilon_step_size == 0 and ep != 0 :
                self.epsilon = (self.epsilon - 0.05 if self.epsilon > 0.1 else 0.1)
            if ep > 3000 and self.epsilon == 0.1:
                self.epsilon = 0.05
            if test_rw > best_score and save_path:
                #self.save("lunar_learned_weights_{0}_{1}_{2}".format(ep, test_rw, self.learning_rate))
                self.save(save_path)
                best_score = test_rw
                #if test_rw >= 200:
                #    break
        outfile.close()
        logfile.close()
        print ('reward is: {0} at iteration {1}'.format(self.test_online(), episodes))
        print ('The weight saved for best moving averate of {0}'.format(best_score))

    """
    This methods recieves the batch_size as the input parameter. It samples those many number of records from the replay buffer.
    For each record in the sample set, following operations are performed -
        1. Call predict on the model (online network) using x
        2. Call predict on the model (online network) using x_new
        3. Call predict on the old_model (target network) using x_new
        4. Choose the best action from step 2, which is the index value of the highest value in the array
        5. Use the index number from 4 to select the q value from step 2
        6. Calcualte the q estimate using target equation - reward if the state is terminal else (reward + gamma* (q value from step 5))
        7. Set the q value to the array returned on step 1 to the index indentified by "action" within the sample record
        8. perform gradient descent

    The weigts are copied to the target network from online network every few episodes as configued in  self.transer_weight_count
    """
    def replay_online(self, ep_st, batch_size=32):
        indexes = np.random.choice(self.min_exploration, size=batch_size)
        batch = [self.exploration_set[itr] for itr in indexes]
        if ep_st % self.transer_weight_count == 0:
            self.model.clone()
        for x,action,reward,x_new,done in batch:
            q_x = self.model.predict(x.reshape(1,8))[0]
            qa = self.model.predict(x_new.reshape(1,8))[0]
            qas_old = self.model.old_model.predict(x_new.reshape(1,8))[0]
            q = reward if done else (reward + self.gamma * qas_old[np.argmax(qa)])
            q_x[action] = q
            self.model.train((x.reshape(1,8),q_x.reshape(1,4)),epoch=1)

    """
    This methods recieves the batch_size as the input parameter. It samples those many number of records from the replay buffer.
    On entire batch, following operations are performed -
        1. Call predict on the model (online network) using x
        2. Call predict on the model (online network) using x_new
        3. Call predict on the old_model (target network) using x_new
        4. Choose the best action from step 2, which is the index value of the highest value in the array
        5. Use the index number from 4 to select the q value from step 2
        6. Calcualte the q estimate using target equation - reward if the state is terminal else (reward + gamma* (q value from step 5))
        7. Set the q value to the array returned on step 1 to the index indentified by "action" within the sample record
        8. perform gradient descent on the whole batch using 32 samples at a time for gradient update.

    The weigts are copied to the target network from online network every few episodes as configued in  self.transer_weight_count
    """
    def replay_batch(self, ep_st, batch_size=32):
        indexes = np.random.choice(len(self.exploration_set), size=batch_size)
        batch = [self.exploration_set[itr] for itr in indexes]
        qvals = np.zeros((batch_size,8))
        qactions = np.zeros((batch_size,4))
        ind = 0
        if ep_st % self.transer_weight_count == 0:
            self.model.clone()
        x_new_batch = np.zeros((batch_size, 8))
        x_batch = np.zeros((batch_size, 8))
        for i in range(len(batch)):
            x,action,reward,x_new,done = batch[i]
            x_new_batch[i] = x_new
            x_batch[i] = x
        q_x = self.model.predict(x_batch)
        qas = self.model.predict(x_new_batch)
        qas_old = self.model.old_model.predict(x_new_batch)
        for i in range(len(batch)):
            x,action,reward,x_new,done = batch[i]
            q = reward if done else reward + self.gamma * qas_old[i][np.argmax(qas[i])] # Choose best action from new model, but, pick the estimate for the action from stationary model
            q_x[i][action] = q
            qvals[ind] = x
            ind+=1
        self.model.train((qvals,q_x),epoch=1, batch_size=32)


    """
    Returns action as per epsilon greedy policy if random_enabled is True. If
    random_enabled is false, uses only the online network to determine the action.
    """
    def getAction(self, x, random_enabled=True):
        a = 0
        if (random.random() < self.epsilon) and random_enabled:
            a = self.env.sample_actionset()
        else:
            qa = self.model.predict(" ".join(x))
            a = np.argmax(qa[0])
        return a

    def save(self, path):
        self.model.save(path)

    """
    Runs bench mark on the weights provided by path variable. By default it runs
     100 iterations and returns rewards from all the episodes
    """
    def test(self, path=None, iteration=100):
        self.env = gym.make('LunarLander-v2')
        self.env = gym.wrappers.Monitor(self.env, directory="temp")
        self.model.load(path)

        for it in range(iteration):
            x = self.env.reset()
            while True:
                #self.render()
                action = self.getAction(x, False)
                x_new, reward, done, _ = self.env.step(action)
                x = x_new
                if done:
                    break
        return self.env.get_episode_rewards()

    """
    Runs bench mark on the weights currently with in the model. By default it runs
     100 iterations and returns rewards from all the episodes
    """
    def test_online(self, iteration=100):
        rewards = []
        for it in range(iteration):
            x = self.env.reset()
            rew = 0
            while True:
                #self.render()
                action = self.getAction(x, False)
                x_new, reward, done, _ = self.env.step(action)
                x = x_new
                rew += reward
                if done:
                    break
            rewards.append(rew)
        return np.average(rewards)

def main():
    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print ('Usage python train_lunarlander.py <learning_rate> <epsilon_step_size> <replay_buffer_size> <episodes> <optional: training_save_file_name>')
        exit()
    lander = LunarLander(learning_rate=float(sys.argv[1]), epsilon_step_size=int(sys.argv[2]), replay_buffer_size=int(sys.argv[3]), )
    lander.act(episodes=int(sys.argv[4]), save_path=sys.argv[5] if len(sys.argv) == 6 else "lunar_learned_weights")
    #lander.save(sys.argv[5] if len(sys.argv) == 6 else "lunar_learned_weights")
    print (lander.epsilon_stat)

if __name__ == '__main__':
    main()
