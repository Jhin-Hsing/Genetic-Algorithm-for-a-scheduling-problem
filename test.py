import random
import numpy as np

# fitness_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# random.shuffle(fitness_values)

# sum_fitness = sum(fitness_values)

# prob_list = [value/sum_fitness for value in fitness_values]

# add_prob_list = [sum(prob_list[:i+1]) for i in range(len(fitness_values))]


# for i in range(len(fitness_values)):
#     fit = round(fitness_values[i],2)
#     prob = round(prob_list[i],2)
#     add = round(add_prob_list[i],2)
#     print(f'fitness:{fit}\t佔比:{prob}\t累積機率:{add}')
# print(f"總和:{sum_fitness}\t總和:{sum(prob_list)}")


# max = sum([c.fitness for c in population])
# selection_probs = [c.fitness/max for c in population]
# return population[npr.choice(len(population), p=selection_probs)]


import matplotlib.pyplot as plt

data = [1, 3, 5, 2, 7, 6, 8, 4]

plt.plot(data)
plt.savefig('./test.jpg')
# plt.show()