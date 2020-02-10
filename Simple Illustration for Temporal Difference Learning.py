import numpy as np


#function that moves the agent left or right
#left = 1
#right = 2
def step(x):
    if action == 1:
        return -1
    if action == 2:
        return 1

#Policy is what maps from states to action
policy = (1, 2)

#First state and last state are equal to zero because V(terminal) = 0
environment = np.array([0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]) 

#Starting point (index)
start = 3

#Finishing Point (index)
goal = len(environment)-1

#current reward
reward = 0

#flag to stop the while loop if reached any terminal state
terminal = False

#accumulating reward
total_rewards = 0

#Learning rate (step size)
lr = 0.3

#current state
state = 0

#discount factor to control the agent (short sighted = 0 or long sigted = 1)
discount_factor = 0.5

#training part
for i in range(400):
    action = np.random.choice(policy, p = [1-environment[start], environment[start]]) #always start at the third index
    state = start #current state (starting state)
    terminal = False
    while not terminal: #as long as the agent did not reach terminal state
        last_state = state
        state += step(action)
        if state == 0:
            terminal = True
            reward = 0
            environment[state+1] = environment[state+1] + lr * (reward + (discount_factor*environment[state]) - environment[state+1])
        elif state == goal:
            reward = 1
            environment[state-1] = environment[state-1] + lr * (reward + (discount_factor*environment[state]) - environment[state-1])
            total_rewards += 1
            terminal = True
        else:
            if state > last_state:
                reward = 0
                environment[state-1] = environment[state-1] + lr * (reward + (discount_factor*environment[state]) - environment[state-1])
            else:
                reward = 0
                environment[state+1] = environment[state+1] + lr * (0 + (discount_factor*environment[state]) - environment[state+1])
        action = np.random.choice(policy, p = [1-environment[state], environment[state]])
        reward = 0
    if i % 100 == 0: #every 100 times it will show environment and total rewards
        print(environment, total_rewards)
