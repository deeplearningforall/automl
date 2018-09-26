from train_lunarlander import LunarLander
import numpy as np
import sys
lander = LunarLander()
iteration = 100
avg_reward = lander.test(sys.argv[1], iteration)
print (avg_reward)
print("Reward for {0} iteration is {1} ".format(iteration, np.average(avg_reward)))
