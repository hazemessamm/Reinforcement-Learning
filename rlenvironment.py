import numpy as np
from PIL import Image
import cv2
from matplotlib.pyplot import style
import pickle
import matplotlib.pyplot as plt
import time

style.use('ggplot')

SIZE = 10
HM_EPISODS = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 350
FOOD_REWARD = 25
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
epsilon = 0.9

start_qtable = None 

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
LEARNING_RATE = 0.1
DISCOUNT = 0.99

d = {1: (255,175,0), 2:(0, 255, 0), 3: (0,0,255)}

class Blop:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    def __str__(self):
        return f'{self.x}, {self.y}'
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    def action(self, choice):
        if choice == 0:
            self.move(x = 1, y = 1)
        elif choice == 1:
            self.move(x = -1, y = -1)
        elif choice == 2:
            self.move(x = -1, y = 1)
        elif choice == 3:
            self.move(x = 1, y = -1)
    def move(self, x = False, y = False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_qtable is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_qtable, 'rb') as f:
        q_table = pickle.load(f)


episode_rewards = []

for episode in range(HM_EPISODS):
    player = Blop()
    food = Blop()
    enemy = Blop()

    if episode % SHOW_EVERY == 0:
        print(f'on #{episode} epsilon: {epsilon}')
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    episode_reward = 0
    for i in range(200):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        player.action(action)

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE + (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            env[food.x][food.y] = d[FOOD_N]
            env[player.x][player.y] = d[PLAYER_N]
            env[enemy.x][enemy.y] = d[ENEMY_N]

            img = Image.fromarray(env, 'RGB')
            img = img.resize((300, 300))
            cv2.imshow('Game!', np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xff == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            episode_reward += reward
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break
        episode_rewards.append(episode_reward)
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'reward {SHOW_EVERY}')
plt.xlabel('episode rewards')
plt.show()


with open(f'qtable-{int(time.time())}.pickle', 'wb') as f:
    pickle.dump(q_table, f)
