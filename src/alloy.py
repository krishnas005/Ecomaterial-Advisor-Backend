import json
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Load the dataset
with open('alloy_dataset.json', 'r') as f:
    dataset = json.load(f)

# Define the properties we want to optimize
properties_to_optimize = ['tensile_strength', 'hardness', 'thermal_resistance', 'density', 'sustainability_score']

# Extract all unique elements from the dataset
all_elements = set()
for material in dataset:
    all_elements.update(material['composition'].keys())

# Convert set to sorted list for consistent ordering
all_elements = sorted(list(all_elements))

# Function to convert composition dict to list
def comp_dict_to_list(comp_dict):
    return [comp_dict.get(element, 0) for element in all_elements]

# Function to convert composition list to dict
def comp_list_to_dict(comp_list):
    return {element: value for element, value in zip(all_elements, comp_list) if value > 0}

# Normalize the composition
def normalize_composition(composition):
    total = sum(composition)
    return [c / total for c in composition]

# Create property estimator using simple weighted average
def estimate_properties(composition):
    properties = {prop: 0 for prop in properties_to_optimize}
    total_weight = 0
    
    for material in dataset:
        weight = sum(min(c1, c2) for c1, c2 in zip(composition, comp_dict_to_list(material['composition'])))
        total_weight += weight
        
        for prop in properties_to_optimize:
            properties[prop] += weight * material['properties'][prop]
    
    if total_weight > 0:
        for prop in properties:
            properties[prop] /= total_weight
    
    return properties

# Fitness function
def evaluate(individual):
    composition = normalize_composition(individual)
    properties = estimate_properties(composition)
    
    fitness = sum((properties[prop] - target_properties[prop])**2 for prop in properties_to_optimize)
    
    # Penalize for using too many elements
    num_elements = sum(1 for c in composition if c > 0.01)
    if num_elements > 5:
        fitness += (num_elements - 5) * 100  # Add penalty for each extra element
    
    return (fitness,)

# Create types
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Toolbox initialization
toolbox = base.Toolbox()

# Modified individual creation
def create_individual():
    ind = [0] * len(all_elements)
    for _ in range(random.randint(2, 5)):  # Randomly select 2 to 5 elements
        ind[random.randint(0, len(all_elements) - 1)] = random.random()
    return creator.Individual(ind)

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main function to run the genetic algorithm
def generate_alloy(target_props, pop_size=100, n_gen=200):
    global target_properties
    target_properties = target_props

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=hof, verbose=True)

    best_ind = tools.selBest(pop, k=1)[0]
    best_comp = normalize_composition(best_ind)
    best_properties = estimate_properties(best_comp)

    return comp_list_to_dict(best_comp), best_properties

# Example usage
if __name__ == "__main__":
    # Save the dataset to a file
    with open('alloy_dataset.json', 'w') as f:
        json.dump(dataset, f)

    # Define target properties
    target_properties = {
        'tensile_strength': 800,
        'hardness': 250,
        'thermal_resistance': 500,
        'density': 7.5,
        'sustainability_score': 7.0
    }

    # Generate new alloy
    new_composition, achieved_properties = generate_alloy(target_properties)

    print("Generated Alloy Composition:")
    for element, percentage in new_composition.items():
        print(f"{element}: {percentage:.2f}%")

    print("\nAchieved Properties:")
    for prop, value in achieved_properties.items():
        print(f"{prop}: {value:.2f}")

    print("\nTarget Properties:")
    for prop, value in target_properties.items():
        print(f"{prop}: {value:.2f}")