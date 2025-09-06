import random
import matplotlib.pyplot as plt
#1
def f(x, y):
    return x**2-abs(y)

def generate_population(size, points):
    return [[random.uniform(points[0][0], points[0][1]),
             random.uniform(points[1][0], points[1][1])]
            for _ in range(size)]

def fitness(population):
    return [f(ind[0], ind[1]) for ind in population]

def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind, fit in zip(population, fitness_values):
        current += fit
        if current > pick:
            return ind

def crossover(parent1, parent2):
    alpha = random.uniform(0.0, 1.0)
    child1 = [alpha * parent1[0] + (1 - alpha) * parent2[0],
              alpha * parent1[1] + (1 - alpha) * parent2[1]]
    child2 = [(1 - alpha) * parent1[0] + alpha * parent2[0],
              (1 - alpha) * parent1[1] + alpha * parent2[1]]
    return child1, child2

def mutate(ind, points, mutation_rate=0.1):
    if random.random() < mutation_rate:
        ind[0] += random.uniform(-0.1, 0.1)
        ind[1] += random.uniform(-0.1, 0.1)
        ind[0] = min(max(ind[0], points[0][0]), points[0][1])
        ind[1] = min(max(ind[1], points[1][0]), points[1][1])
    return ind


def genetic_algorithm(points, pop_size=50, generations=1000, mutation_rate=1):
    population = generate_population(pop_size, points)
    best_values = []
    best_indexes = []
    
    for i in range(generations):
        fitness_values = fitness(population)
        new_population = []
        
        
        best_idx = fitness_values.index(max(fitness_values))
        best_values.append(fitness_values[best_idx])
        best_indexes.append(population[best_idx])
        
        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(population, fitness_values)
            parent2 = roulette_wheel_selection(population, fitness_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, points, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, points, mutation_rate))

        population = new_population
        print("Найкраще значення на ітерації ",i,":", best_values[i])
        print("Найкраще значення на ітерації ",i,":", best_indexes[i])
    
    fitness_values = fitness(population)
    best_idx = fitness_values.index(max(fitness_values))
    best_ind = population[best_idx]
    

    return best_ind, f(best_ind[0], best_ind[1]), best_values,best_indexes

# --- Запуск ---
if __name__ == "__main__":
    points = [[-4, 2], [1, 5]]
    best_ind, best_value, best_values,best_indexes = genetic_algorithm(points)

    print("Значення екстремуму:", max(best_values)) 
    print("Екстремум в точці:",best_indexes[best_values.index(max(best_values))]) 

    # График прогресса 
    plt.figure(figsize=(10,6), dpi=400) 
    plt.plot(best_values, linewidth=1) 
    plt.xlabel("Ітерація") 
    plt.ylabel("Найкраще значення функції") 
    plt.title("Генетичний алгоритм") 
    plt.grid(True) 
    plt.savefig("progress1.png") 