import random


a = []

for i in range(15):
      rnd = random.randint(0,15)
      while(rnd in a):
            rnd = random.randint(0,15)

      a.append(rnd)
print(a)