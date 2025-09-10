import random
import matplotlib.pyplot as plt

cords = [[-4, 2], [1, 5]]
mutate_chance = 0.3
iterations = 500
count_of_pop = 50
bits = 16

def function(x, y):
    return x**2 - abs(y)

def fitness(pop):
    return [function(ind[0], ind[1]) for ind in pop]

def roulette_function(pop, fit_values):
    fit_sum = sum(fit_values)
    winner = random.uniform(0, fit_sum)
    current = 0
    for i in range(len(pop)):
        current += fit_values[i]
        if current >= winner:
            return pop[i]
    return pop[-1]

def mutate(child, mutate_chance=0.1):
    if random.random() < mutate_chance:
        child[0] += random.uniform(-0.1, 0.1)
        child[1] += random.uniform(-0.1, 0.1)
        child[0] = min(max(child[0], cords[0][0]), cords[0][1])
        child[1] = min(max(child[1], cords[1][0]), cords[1][1])
    return child

def pop_generator(size, cords):
    return [[random.uniform(cords[0][0], cords[0][1]),
             random.uniform(cords[1][0], cords[1][1])] for _ in range(size)]

def encode(value, bound, bits=16):
    min_v, max_v = bound
    norm = (value - min_v) / (max_v - min_v)
    int_val = int(norm * (2**bits - 1))
    return format(int_val, f'0{bits}b')

def decode(bits_str, bound, bits=16):
    min_v, max_v = bound
    int_val = int(bits_str, 2)
    norm = int_val / (2**bits - 1)
    return min_v + norm * (max_v - min_v)

def crossover(parent1, parent2):
    p1x, p1y = encode(parent1[0], cords[0], bits), encode(parent1[1], cords[1], bits)
    p2x, p2y = encode(parent2[0], cords[0], bits), encode(parent2[1], cords[1], bits)

    point_x = random.randint(1, bits - 1)
    point_y = random.randint(1, bits - 1)

    c1x = p1x[:point_x] + p2x[point_x:]
    c2x = p2x[:point_x] + p1x[point_x:]

    c1y = p1y[:point_y] + p2y[point_y:]
    c2y = p2y[:point_y] + p1y[point_y:]

    child1 = [decode(c1x, cords[0], bits), decode(c1y, cords[1], bits)]
    child2 = [decode(c2x, cords[0], bits), decode(c2y, cords[1], bits)]

    return child1, child2

def save_picture(best_values):
    plt.figure(figsize=(10,6), dpi=300)
    plt.plot(best_values, linewidth=1)
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення")
    plt.title("Прогрес ГА")
    plt.grid(True)
    plt.savefig("evolution.png")

def main():
    population = pop_generator(count_of_pop, cords)
    best_values = []
    best_points = []

    for gen in range(iterations):
        new_population = []
        fit_values = fitness(population)

        best_idx = fit_values.index(max(fit_values))
        best_values.append(fit_values[best_idx])
        best_points.append(population[best_idx])

        while len(new_population) < count_of_pop:
            parent1 = roulette_function(population, fit_values)
            parent2 = roulette_function(population, fit_values)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutate_chance))
            if len(new_population) < count_of_pop:
                new_population.append(mutate(child2, mutate_chance))

        population = new_population

    fit_values = fitness(population)
    best_idx = fit_values.index(max(fit_values))
    best_ind = population[best_idx]

    print("Максимум", function(best_ind[0], best_ind[1]))
    print("Точка:", best_ind)

    save_picture(best_values)

if __name__ == "__main__":
    main()
