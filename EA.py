import numpy as np
class EAResult:
    def __init__(self) -> None:
        self.best_solution = None
        self.best_fitness = 0
        self.fitness_history = []
        self.mr_history = []
def gaussian_mutation(x, mutation_rate):
    return x + np.random.standard_normal(len(x)) * mutation_rate

def simple_opoea(chromosome_size, self_adaptative_mutation_rate, num_gens, fitness_function):
    ea_result = EAResult()
    best_solution = (np.random.rand(chromosome_size) - 0.5)/2
    best_fitness = fitness_function(best_solution)
    mutation_rate = np.ones(chromosome_size)/chromosome_size**2
    for i in range(num_gens):
        new_mutation_rate = mutation_rate * np.exp(np.random.standard_normal(chromosome_size) * self_adaptative_mutation_rate)
        new_solution = gaussian_mutation(best_solution, new_mutation_rate)
        new_fitness = fitness_function(new_solution)
        
        if new_fitness >= best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness
            mutation_rate = new_mutation_rate
        
        if i%100 == 0:
            ea_result.fitness_history.append(best_fitness)
            ea_result.mr_history.append(mutation_rate)
    
    ea_result.best_solution = best_solution
    ea_result.best_fitness = best_fitness
    return ea_result

