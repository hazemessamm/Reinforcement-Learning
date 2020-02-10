import numpy as np

def step(x):
    if action == 1:
        return -1
    if action == 2:
        return 1


policy = (1, 2)

environment = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0])

start = 3

reward = 0

terminal = False

total_rewards = 0

lr = 0.3

j = 0

for i in range(50):
    action = np.random.choice(policy, p = [1-environment[start], environment[start]])
    j = start
    terminal = False
    while not terminal:
        last_action = j
        j += step(action)
        if j == 0:
            terminal = True
            environment[j+1] = environment[j+1] + lr * (0 + (0.5*environment[j]) - environment[j+1])
        elif j == 6:
            environment[j-1] = environment[j-1] + lr * (1 + (0.5*environment[j]) - environment[j-1])
            total_rewards += 1
            terminal = True
        else:
            if j > last_action:
                environment[j-1] = environment[j-1] + lr * (0 + (0.5*environment[j]) - environment[j-1])
            else:
                environment[j+1] = environment[j+1] + lr * (0 + (0.5*environment[j]) - environment[j+1])
        action = np.random.choice(policy, p = [1-environment[j], environment[j]])
    if i % 100 == 0:
        print(environment, total_rewards)


        
