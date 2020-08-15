import gym
import numpy as np
import baselines.baseline_logger as bl
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

GYM_ENV = 'CartPole-v1'
NUM_RUNS = 30
LOGGING_DIRECTORY = 'results/experiment_4_new_baselines'
EXPERIMENT_NAME = 'PPO'

def main():
    logger = bl.BaseLineLogger(EXPERIMENT_NAME, GYM_ENV, LOGGING_DIRECTORY)
    for run_idx in range(NUM_RUNS):
        run(logger)

    logger.end_experiment()


def run(logger: bl.BaseLineLogger):
    env = gym.make('CartPole-v1')
    model = PPO2(MlpPolicy, env,
                n_steps=32,
                nminibatches=1,
                lam=0.8,
                gamma=0.98,
                noptepochs=20,
                ent_coef=0.0,
                learning_rate=0.001,
                cliprange = 0.2,
                verbose=1)

    for batch_idx in range(50):
        model.learn(total_timesteps=2000)

        test_fitness = test_algorithm(model)
        total_timesteps = (batch_idx + 1) * 2000
        logger.save_fitnesses(test_fitness, total_timesteps, batch_idx)
        logger.print_data()

        if test_fitness == 500.0:
            break

    logger.end_run()



def test_algorithm(model) -> float:
    """
    Runs a test set of 100 rollouts on the current model.
    No data is stored for learning. At test time, the actions are
    ensembled together.

    Returns: the mean score/fitness of the 100 runs
    """
    env = gym.make(GYM_ENV)
    fitnesses = []

    for _ in range(100):
        fitness = 0
        obs = env.reset()
        while True:
            action, _ = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            fitness += rewards
            if done:
                break
        fitnesses.append(fitness)
    return np.mean(fitnesses)


if __name__ == '__main__':
    main()