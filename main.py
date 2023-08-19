import love_sim_class as sim
import time
from tqdm import tqdm

# Simulation parameters
TIME_STEPS = 300
NUMBER_OF_AGENTS = 5000

def main(number_of_intervals, pop_size):
    amorous_entities = sim.population(probabilistic_sampling_func, pop_size)
    start_time = time()
    for t in tqdm(range(number_of_intervals)):
        amorous_entities.full_time_interval()
    print("\nrun complete")
    print(time() - start_time)
    return amorous_entities

if __name__ == "__main__":
    print("Commencing simulation at", time.strftime("%H:%M:%S", t))
    main(TIME_STEPS, NUMBER_OF_AGENTS)
    
  
