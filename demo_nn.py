import gym
import torch
import json
import sys
import pickle
import numpy as np

try:
    import safety_gym
except ModuleNotFoundError:
    pass

nn_path = sys.argv[1]

nn_dict = pickle.load(open(nn_path, 'rb'))

ENVIRONMENT = nn_dict['env']
GYM_ENV = gym.make(ENVIRONMENT)


#For performance
if ENVIRONMENT == 'CartPole-v1':
    ENV_SWITCHER = 0
else:
    ENV_SWITCHER = 1


def demo_best_net(nn):
    obs = GYM_ENV.reset()
    curr_obs = obs
    fitness = 0
    while True:
        GYM_ENV.render()

        action = nn.forward(torch.from_numpy(obs).float())
        print(action)

        if ENV_SWITCHER == 0:
            #argmax
            action = action.max(0)[1].item()
        elif ENV_SWITCHER == 1:
            action = action.detach().numpy()
            #action -= 0.5
            #action *= 2

        #print(action)
        obs, reward, done, hazards = GYM_ENV.step(action)
        print(reward) 
        fitness += reward
        print(action)
        print('\n')

        obs_diff = np.sum(np.absolute(obs-curr_obs))
        curr_obs = obs
        
        #print(obs_diff)

        if done:
            break
            
    print(f"Demo Fitness: {fitness}")

print(GYM_ENV.action_space)
print(GYM_ENV.observation_space)
print(GYM_ENV.action_space.high)
print(GYM_ENV.action_space.low)
print(GYM_ENV.action_space.sample())

demo_best_net(nn_dict['nn'])