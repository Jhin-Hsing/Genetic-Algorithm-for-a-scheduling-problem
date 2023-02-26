import random

# 初始化種群
def init_population(pop_size, chrom_length, min_value, max_value):
    population = []
    for i in range(pop_size):
        chrom = [random.uniform(min_value, max_value) for j in range(chrom_length)]
        population.append(chrom)
    return population

# 計算適應度
def calc_fitness(chrom):
    fitness = 0
    for i in range(len(chrom)):
        fitness += chrom[i]**2
    return fitness

# 選擇
def selection(population):
    fitness_list = [calc_fitness(chrom) for chrom in population]
    sum_fitness = sum(fitness_list)
    # 輪盤選擇
    selected_pop = []
    for i in range(len(population)):
        pick = random.uniform(0, sum_fitness)
        current = 0
        for j in range(len(population)):
            current += fitness_list[j]
            if current > pick:
                selected_pop.append(population[j])
                break
    return selected_pop

# 交配
def crossover(p1, p2, crossover_rate):
    # 以一定機率進行交配
    if random.random() < crossover_rate:
        chrom_length = len(p1)
        # 隨機選擇一個交配點
        crossover_point = random.randint(1, chrom_length - 1)
        # 交換基因
        offspring1 = p1[:crossover_point] + p2[crossover_point:]
        offspring2 = p2[:crossover_point] + p1[crossover_point:]
    else:
        # 如果不進行交配，則直接複製父代染色體
        offspring1 = p1[:]
        offspring2 = p2[:]
    return offspring1, offspring2

# 突變
def mutate(individual, mutation_rate, min_value, max_value):
    for i in range(len(individual)):
        # 以一定機率對基因進行突變
        if random.random() < mutation_rate:
            # 隨機生成一個突變值
            mutation_value = random.uniform(min_value, max_value)
            # 用突變值替換原基因
            individual[i] = mutation_value
    return individual



# 初始化參數
POP_SIZE = 50
CHROM_LENGTH = 10
MIN_VALUE = -5.0
MAX_VALUE = 5.0
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
MAX_GENERATION = 100

# 初始化種群
population = init_population(POP_SIZE, CHROM_LENGTH, MIN_VALUE, MAX_VALUE)

# 進行世代演化
for generation in range(MAX_GENERATION):
    # 選擇
    selected_pop = selection(population)

    # 交配
    offspring = []
    for i in range(0, POP_SIZE-1, 2):
        p1 = selected_pop[i]
        p2 = selected




def genetic_algorithm(population_size, num_generations, selection_func, crossover_func, mutation_func):
    # 初始化種群
    population = init_population(population_size)

    # 進行世代演化
    for generation in range(num_generations):

        # 選擇
        mating_pool = selection_func(population)

        # 交配
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = mating_pool[i], mating_pool[i+1]
            if random.random() < 0.8:
                c1, c2 = crossover_func(p1, p2)
            else:
                c1, c2 = p1, p2
            offspring.extend([c1, c2])

        # 突變
        for i in range(len(offspring)):
            if random.random() < 0.2:
                offspring[i] = mutation_func(offspring[i])

        # 計算適應度
        fitness_population = [fitness(individual) for individual in population]
        best_fitness = max(fitness_population)

        # 選擇下一代種群
        population = select_next_generation(population, offspring, fitness_population)

    return population