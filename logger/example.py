from evo_logger import EvoLogger
import random

num_runs = 5
num_epochs = 5


output_name = 'test_name'
output_folder = None # If using, leave the last / off. Example "checkpoints/experiment1"
use_pickle = True # Will try to pickle the results. Can be used to store more advanced data. 

logger = EvoLogger(output_name, output_folder, use_pickle=True)

for run_idx in range(num_runs):
    num_gens = 0
    timesteps = 0
    gen = 0
    for epoch in range(num_epochs):

        # Model train here

        # Test model here

        test_fit = random.uniform(0, 500) # "Test Results"
        
        logger.save_sample(test_fit, timesteps, gen) # These params are required, other optional ones if needed. 
        logger.print_data() # Optional

        num_gens += 1
        timesteps += 1000
        gen += 10
    
    logger.end_run() # Required At the end of each run to reset vars.  

logger.end_experiment() # Saves the logs. 