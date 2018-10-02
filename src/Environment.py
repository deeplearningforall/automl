# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:37:11 2018

@author: Himanshu
"""
import random

class Environment:
    def __init__(self, agent_utility):
        self.accuracy = 0.0
        self.loss = 0.0
        self.validation_accuracy = 0.0
        self.validation_loss = 0.0
        self.reward_history = []
        self.action_space = ['LSTM_RST', 'LSTM_RSF', 'LSTM_RST_BI', \
                             'LSTM_RSF_BI', 'GRU_RST', 'GRU_RSF', \
                             'GRU_RST_BI', 'GRU_RSF_BI', \
                             'Dense_Sigmoid', 'Dense_Tanh', \
                             'Dense_Relu', 'MaxPool1D', 'Conv1D', \
                             'Flatten']
    
    def set_action_utility(self,agent_utility):
        self.agent_utility = agent_utility
    
    def sample_actionset(self):
        return random.choice(self.action_space)
    
    def reset_stats(self):
        self.accuracy = 0.0
        self.loss = 0.0
        self.validation_accuracy = 0.0
        self.validation_loss = 0.0
        return self.agent_utility.model_layers

    def set(self,action):
        if(self.agent_utility):
            self.agent_utility.perform_operation(action)
            history = self.agent_utility.model.fit(self.agent_utility.x, self.agent_utility.y, batch_size=128, epochs=4, validation_split=0.2)
            reward = 0
            if(history.history["val_loss"] < self.validation_loss):
                reward = 1
                self.validation_loss = history.history["val_loss"]
            else:
                reward = -1
            self.reward_history.apppend(reward)
            if(self.reward_history[-3:] == [-1, -1, -1]):
                return self.agent_utility.model_layers, reward, True
            else:
                return self.agent_utility.model_layers, reward, False
        else:
            raise Exception("Please set aagent utility object first")