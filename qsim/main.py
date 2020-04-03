from Quad_demo import Quadrotor
import numpy as np
# from PIL import Image
# import cv2 as cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import numpy as np
import csv



show_animation = True
from mpl_toolkits.mplot3d import Axes3D
x_pos = -5
CONSTANT_Z=5
y_pos = -5
z_pos = 5
x_vel = 0
y_vel = 0
z_vel = 0
x_acc = 0
y_acc = 0
z_acc = 0
roll = 0
pitch = 0
yaw = 0
roll_vel = 0
pitch_vel = 0
yaw_vel = 0

des_yaw = 0

dt = 0.1
t = 0



style.use("ggplot")

SIZE = 10
HM_EPISODES = 50000
MOVE_PENALTY = 2
ENEMY_PENALTY = 30
FOOD_REWARD = 70
goal_coord=(5,5)
epsilon = 0.9
EPS_DECAY = 0.9998

SHOW_EVERY = 300
# pickle_in = open("qtable-1583896702.pickle","rb")
# example_dict = pickle.load(pickle_in)
 # or filename

LEARNING_RATE = 0.1
DISCOUNT = 0.90

PLAYER_N = 1

q = Quadrotor(x=x_pos, y=y_pos, z=z_pos,x1=x_pos+3, y1=y_pos+5, z1=z_pos,   roll=roll, pitch=pitch, yaw=yaw, size=1)
acs=[]
obse =[]
class Blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0, SIZE)
        self.z = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y},{self.z}"
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y, self.z-other.z)
    def action(self,choice):
        if choice == 0:
            self.move(x=1,y=0,z=0)
        elif choice == 1:
            self.move(x=-1,y=0,z=0)
        elif choice == 2:
            self.move(x=0,y=1,z=0)
        elif choice == 3:
            self.move(x=0,y=-1,z=0)
        elif choice == 4:
            self.move(x=0,y=0,z=1)
        elif choice == 5:
            self.move(x=0,y=0,z=-1)
        elif choice == 6:
            self.move(x=0, y=0, z=0)
 #doubtful
        pass
    def move(self,x=False,y=False,z=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
        if not z:
            self.z += np.random.randint(-1, 2)
        else:
            self.z += z
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE - 1:
            self.x = SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE - 1:
            self.y = SIZE - 1
        if self.z < 0:
            self.z = 0
        elif self.z > SIZE - 1:
            self.z = SIZE - 1



start_q_table = None
q_table={}
# start_q_table= "qtable-1585877759.pickle"
# with open(start_q_table, "rb") as f:
#     q_table = pickle.load(f)
if start_q_table is None:

    for x1 in range(-SIZE,SIZE):
        for y1 in range(-SIZE , SIZE):
            for z1 in range(-SIZE,SIZE):
                q_table[(x1,y1,z1)] = [np.random.uniform(0,5) for i in range(7)]

else:
    with open(start_q_table, "rb") as f:
        print("-----------------Loaded q_table--------------------")
        q_table = pickle.load(f)
episode_rewards=[]
food = Blob()
food.x = 4 #Goal position
food.y = 4
food.z = 4 #Goal position
food.roll = 0
prev=[0,0,0,0]
MAX_REWARD = 850
for episodes in range(HM_EPISODES):
    player1 = Blob()
    player2 = Blob()
    reward = 0

    if episodes % SHOW_EVERY == 0:
        print(f"on #{episodes}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else :
        show = False
    episode_reward = 0
    for i in range(200):
        obs = (player1-player2)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action= np.random.randint(0,7)

        player1.action(action)
        player2.action(action)
        acs.append(action)
        obse.append(obs)
        ######################
        """
        EDIT THE SECTION BELOW
        REWARD SECTION BEGIN
        """
        if player1.x == food.x and player1.y == food.y and player1.z == food.z:
            reward += FOOD_REWARD/2
        elif player1.x == food.x:
            reward += 2
        elif player1.z == food.z:
            reward += 2
        elif player1.y == food.y:
            reward += 2
        if player2.x == food.x and player2.y == food.y+4 and player2.z == food.z:
            reward += FOOD_REWARD/2
        elif player2.x == food.x:
            reward += 2
        elif player2.z == food.z:
            reward += 2
        elif player2.y == food.y+4:
            reward += 2
        else:
            reward = -MOVE_PENALTY
        '''
        REWARD SECTION END  
        No need to edit anything else
        '''

     ###########################
        new_obs = (player1-player2)
        prev = new_obs
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q)
        q_table[obs][action] = new_q
        # if episodes % 1000==0:
        #      q.update_pose(x=player1.x, y=player1.y, z=player1.z, x1=player2.x, y1=player2.y, z1=player2.z,roll=0, pitch=0, yaw=0)
        episode_reward += reward
        if reward == 85:
            break
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY,mode="valid")

np.savetxt("action.csv", acs, delimiter=",", fmt='%d', header="Action")
np.savetxt("observation.csv", obse, delimiter=",", fmt='%d', header="Observation")
# with open('QVALUE.csv', 'w') as f:
#     for key in q_table.keys():
#         f.write("%s,%s\n"%((key[0],key[1],key[2],key[3]),q_table[key]))
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)

plt.figure()
plt.plot([i for i in range(len(moving_avg))],moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
plt.savefig("drone.png")
