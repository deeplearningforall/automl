# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 13:37:11 2018

@author: Himanshu
"""
import random

class Environment:
    def __init__(self):
        self.accuracy = 0.0
        self.loss = 10000.0
        self.validation_accuracy = 0.0
        self.validation_loss = 10000.0
        self.reward_history = []
        self.action_space = ['add:LSTM_RST', 'add:LSTM_RSF', 'add:LSTM_RST_BI', \
                             'add:LSTM_RSF_BI', 'add:GRU_RST', 'add:GRU_RSF', \
                             'add:GRU_RST_BI', 'add:GRU_RSF_BI', \
                             'add:Dense_Sigmoid', 'add:Dense_Tanh', \
                             'add:Dense_Relu', 'add:MaxPool1D', 'add:Conv1D', \
                             'add:Flatten']
    
    def set_action_utility(self,agent_utility):
        self.agent_utility = agent_utility
    
    def sample_actionset(self):
        return random.choice(self.action_space)
    
    def reset_stats(self):
        self.accuracy = 0.0
        self.loss = 10000.0
        self.validation_accuracy = 0.0
        self.validation_loss = 10000.0
        return self.agent_utility.model_layers

    def step(self,action):
        if(self.agent_utility):
            try:
                self.agent_utility.perform_operation(action)
            except Exception as e:
                print(str(e))
                return self.agent_utility.model_layers, -1, True
                
            history = self.agent_utility.model.fit(self.agent_utility.x, self.agent_utility.y, batch_size=512, epochs=1, validation_split=0.2)
            reward = 0
            if(history.history["val_loss"][-1] < self.validation_loss):
                reward = 1
                self.validation_loss = history.history["val_loss"]
            else:
                reward = 1
            self.reward_history.append(reward)
            if(self.reward_history[-3:] == [-1, -1, -1]):
                return self.agent_utility.model_layers, reward, True
            else:
                return self.agent_utility.model_layers, reward, False
        else:
            raise Exception("Please set aagent utility object first")