import random
from typing import List, Tuple
import matplotlib.pyplot as plt

# Цільова/фітнес-функція
def f(x: float, y: float) -> float:
    return x**2 - y**2

# Параметри діапазонів і довжини хромосом
ax, bx = 0.0, 4.0
cy, dy = 1.0, 2.0
Lx, Ly = 6, 4   # 6 біт для x, 4 біти для y

def encode_x(x: float) -> int:
    k = round((x - ax) * (2**Lx - 1) / (bx - ax))
    return max(0, min(2**Lx - 1, k))

def encode_y(y: float) -> int:
    k = round((y - cy) * (2**Ly - 1) / (dy - cy))
    return max(0, min(2**Ly - 1, k))

def decode_x(kx: int) -> float:
    return ax + kx * (bx - ax) / (2**Lx - 1)

def decode_y(ky: int) -> float:
    return cy + ky * (dy - cy) / (2**Ly - 1)

def to_bits(k: int, L: int) -> str:
    return format(k, f'0{L}b')

def from_bits(bits: str) -> int:
    return int(bits, 2)

# Генерація початкової популяції методом «ковдри» (6 особин)
def init_population() -> List[str]:
    seeds = [(0.0, 2.0), (0.0, 1.0), (2.0, 1.5), (4.0, 2.0), (3.0, 1.0), (1.0, 1.5)]
    pop = []
    for x, y in seeds:
        kx, ky = encode_x(x), encode_y(y)
        chrom = to_bits(kx, Lx) + to_bits(ky, Ly)  # 10 біт
        pop.append(chrom)
    return pop

def decode_chrom(chrom: str) -> Tuple[float, float]:
    kx = from_bits(chrom[:Lx])
    ky = from_bits(chrom[Lx:])
    return decode_x(kx), decode_y(ky)

def fitness(chrom: str) -> float:
    x, y = decode_chrom(chrom)
    return f(x, y)

# Рулетка зі зсувом (щоб уникнути від’ємних)
def roulette_select(pop: List[str]) -> str:
    fits = [fitness(ch) for ch in pop]
    min_fit = min(fits)
    shifted = [(ft - min_fit) + 1e-6 for ft in fits]
    total = sum(shifted)
    r = random.uniform(0.0, total)
    c = 0.0
    for ch, w in zip(pop, shifted):
        c += w
        if c >= r:
            return ch
    return pop[-1]

# Одноточковий кросовер по 10-бітній хромосомі
def crossover(p1: str, p2: str, pc: float = 0.9) -> Tuple[str, str]:
    if random.random() > pc:
        return p1, p2
    point = random.randint(1, Lx + Ly - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

# Побітова мутація
def mutate(ch: str, pm: float = 0.02) -> str:
    bits = list(ch)
    for i in range(len(bits)):
        if random.random() < pm:
            bits[i] = '1' if bits[i] == '0' else '0'
    # проєктуємо назад у допустимий діапазон (на випадок округлень це зайве, але хай буде)
    kx = from_bits(''.join(bits[:Lx])); ky = from_bits(''.join(bits[Lx:]))
    kx = max(0, min(2**Lx - 1, kx))
    ky = max(0, min(2**Ly - 1, ky))
    return to_bits(kx, Lx) + to_bits(ky, Ly)

def genetic_algorithm(generations=500, N=6, pc=0.9, pm=0.02):
    pop = init_population()
    best_hist = []
    elite = max(pop, key=fitness)

    for _ in range(generations):
        # еліта
        new_pop = [elite]
        # добираємо решту
        while len(new_pop) < N:
            p1 = roulette_select(pop)
            p2 = roulette_select(pop)
            c1, c2 = crossover(p1, p2, pc)
            c1, c2 = mutate(c1, pm), mutate(c2, pm)
            new_pop.extend([c1, c2])
        pop = new_pop[:N]
        elite = max(pop, key=fitness)
        best_hist.append(fitness(elite))

    x_best, y_best = decode_chrom(elite)

    # округлення до потрібної звітної точності (0.1)
    x_rep = round(x_best, 1)
    y_rep = round(y_best, 1)
    M = f(x_rep, y_rep)
    return (x_rep, y_rep, M, best_hist)

if __name__ == "__main__":
    random.seed(0)
    x_star, y_star, M, hist = genetic_algorithm(generations=60)

    print("Найкраща точка (≈0.1):", (x_star, y_star))
    print("Значення M:", M)

    # --- Графік прогресу ---
    plt.figure(figsize=(10,6), dpi=200)
    plt.plot(range(1, len(hist)+1), hist, linewidth=1.5, marker="o", markersize=3)
    plt.xlabel("Ітерація (покоління)")
    plt.ylabel("Найкраще значення функції")
    plt.title("Прогрес генетичного алгоритму")
    plt.grid(True)

    # Підписати кількість ітерацій на осі X
    plt.xticks(range(0, len(hist)+1, max(1, len(hist)//10)))  

    plt.savefig("progress2.png")
