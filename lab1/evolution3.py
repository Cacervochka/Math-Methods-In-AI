import random
import matplotlib.pyplot as plt

# Цільова функція
def f(x, y):
    return x**2 - abs(y)

# Генерація початкової популяції
def generate_population(size, bounds):
    return [[random.uniform(bounds[0][0], bounds[0][1]),
             random.uniform(bounds[1][0], bounds[1][1])]
            for _ in range(size)]

# Оцінка пристосованості (fitness)
def evaluate_population(population):
    return [(ind, f(ind[0], ind[1])) for ind in population]

# Рулетка (з урахуванням від’ємних значень)
def roulette_wheel_selection(pop_with_fitness):
    min_fit = min(f for _, f in pop_with_fitness)
    shifted_fitness = [(ind, fit - min_fit + 1e-6) for ind, fit in pop_with_fitness]

    total_fitness = sum(fit for _, fit in shifted_fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind, fit in shifted_fitness:
        current += fit
        if current > pick:
            return ind

# Кросовер (BLX-α або звичайний)
def crossover(parent1, parent2):
    alpha = random.random()
    child1 = [alpha * parent1[0] + (1 - alpha) * parent2[0],
              alpha * parent1[1] + (1 - alpha) * parent2[1]]
    child2 = [(1 - alpha) * parent1[0] + alpha * parent2[0],
              (1 - alpha) * parent1[1] + alpha * parent2[1]]
    return child1, child2

# Мутація
def mutate(ind, bounds, mutation_rate=0.1):
    if random.random() < mutation_rate:
        ind[0] += random.uniform(-0.1, 0.1)
        ind[1] += random.uniform(-0.1, 0.1)
        ind[0] = min(max(ind[0], bounds[0][0]), bounds[0][1])
        ind[1] = min(max(ind[1], bounds[1][0]), bounds[1][1])
    return ind

# Генетичний алгоритм
def genetic_algorithm(bounds, pop_size=50, generations=500, mutation_rate=0.2):
    population = generate_population(pop_size, bounds)
    history = []

    for gen in range(generations):
        pop_with_fitness = evaluate_population(population)

        # Елітизм: зберігаємо найкращого
        best_ind, best_fit = max(pop_with_fitness, key=lambda x: x[1])
        history.append(best_fit)

        new_population = [best_ind]  # переносимо найкращого

        while len(new_population) < pop_size:
            parent1 = roulette_wheel_selection(pop_with_fitness)
            parent2 = roulette_wheel_selection(pop_with_fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, bounds, mutation_rate))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, bounds, mutation_rate))

        population = new_population

    # Фінальний найкращий
    pop_with_fitness = evaluate_population(population)
    best_ind, best_fit = max(pop_with_fitness, key=lambda x: x[1])
    return best_ind, best_fit, history

# --- Запуск ---
if __name__ == "__main__":
    bounds = [[-4, 2], [1, 5]]
    best_ind, best_value, history = genetic_algorithm(bounds)

    print("Екстремум в точці:", best_ind)
    print("Значення функції:", best_value)

    # Графік прогресу
    plt.figure(figsize=(10,6), dpi=200)
    plt.plot(range(1, len(history)+1), history, linewidth=1.5, marker="o", markersize=2)
    plt.xlabel("Ітерація")
    plt.ylabel("Найкраще значення функції")
    plt.title(f"Генетичний алгоритм (Generations={len(history)})")
    plt.grid(True)
    plt.savefig("progress3.png")
