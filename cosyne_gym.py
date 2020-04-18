from cosyne_base.cosyne import Cosyne as cs
import gym
import torch
import json
import sys
import pickle

try:
    import safety_gym
except ModuleNotFoundError:
    pass


config_path = sys.argv[1]

with open(config_path, 'r') as config_file:
    config_dict = json.load(config_file)

ENVIRONMENT = config_dict['cosyne']['env']
GYM_ENV = gym.make(ENVIRONMENT)


#For performance
if ENVIRONMENT == 'CartPole-v1':
    ENV_SWITCHER = 0
    REPEAT_COUNT = 5
    MIN_SCORE = 500.0
else:
    ENV_SWITCHER = 1
    REPEAT_COUNT = 1
    MIN_SCORE = 999999


def eval_gym(nn):
    fitness = 0.01
    
    for i in range(REPEAT_COUNT):
        obs = GYM_ENV.reset()
        if fitness > MIN_SCORE * i:
            while True:       
                action = nn.forward(torch.from_numpy(obs).float())

                if ENV_SWITCHER == 0:
                    #argmax
                    action = action.max(0)[1].item()
                elif ENV_SWITCHER == 1:
                    action = action.detach().numpy()
                    #action -= 0.5
                    #action *= 2

                obs, reward, done, hazards = GYM_ENV.step(action) 
                fitness += reward
                
                if done:
                    break
            
    return fitness

def demo_best_net(nn):
    obs = GYM_ENV.reset()
    fitness = 0
    while True:
        GYM_ENV.render()
        action = nn.forward(torch.from_numpy(obs).float())

        if ENV_SWITCHER == 0:
                #argmax
            action = action.max(0)[1].item()
        elif ENV_SWITCHER == 1:
            action = action.detach().numpy()
            #action -= 0.5
            #action *= 2

        obs, reward, done, hazards = GYM_ENV.step(action) 
        fitness += reward
        
        if done:
            break
            
    print(f"Demo Fitness: {fitness}")



print(GYM_ENV.action_space)
print(GYM_ENV.observation_space)
print(GYM_ENV.action_space.sample())


cosyne = cs(config_dict)
cosyne.run(eval_gym)
print(cosyne.best_fitness)
demo_best_net(cosyne.best_nn)
GYM_ENV.close()
cosyne.export_data()