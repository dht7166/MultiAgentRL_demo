import gym
from gym import spaces
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm

cm = ListedColormap(['w','black','blue','red','green','yellow'])
norm = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5,5.5],ncolors=6)
PATH = 0
WALL = 1
PLAYER = [2,3,4,5] # 4 players total

# action construct
NOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Reward
CRASH = -1
REACH_OBJ = 1
STEP = -1

class World(gym.Env):
    def __init__(self,world_size = 10,agent_num = 4,custom_map = None):
        self.size = world_size
        self.world = custom_map
        if self.world is None:
            self.world = np.zeros((world_size,world_size))
        self.player = []
        self.objective = []
        self.dist = int((world_size-1)/(agent_num-1))
        for i in range(agent_num):
            self.player.append((world_size-1,i*self.dist))
            self.objective.append((0,i*self.dist))
        self.objective.reverse()

        self.observation_space = [spaces.Box(low=0, high=PLAYER[0], shape=(world_size,world_size), dtype=int)
                                  for _ in range(agent_num)]
        self.action_space = [spaces.Discrete(5) for _ in range(agent_num)]
        self.render_holder = plt.imshow(np.copy(self.world), cmap=cm, norm=norm)


    def reset(self):
        for i in range(len(self.player)):
            self.player[i] = (self.size-1,i*self.dist)
        return self.get_observation()


    def add_random_obstacle(self):
        self.world[int(self.size/2)] = np.random.randint(WALL+1,size=self.size)
        self.world[int(self.size/2),np.random.randint(self.size)] = PATH


    def render(self):
        world = np.copy(self.world)
        for i in range(len(self.player)):
            x,y = self.player[i]
            ox,oy = self.objective[i]
            world[x,y] = PLAYER[i]
            world[ox,oy] = PLAYER[i]
        self.render_holder.set_data(world)
        plt.pause(0.1)
        plt.draw()

    def get_render(self):
        world = np.copy(self.world)
        for i in range(len(self.player)):
            x, y = self.player[i]
            ox, oy = self.objective[i]
            world[x, y] = PLAYER[i]
            world[ox, oy] = PLAYER[i]
        return world

    def get_observation(self):
        observation = []

        for i in range(len(self.player)):
            world = np.copy(self.world)
            for j in range(len(self.player)):
                x, y = self.player[j]
                # ox, oy = self.objective[j]
                world[x, y] = PLAYER[1]
                # world[ox, oy] = PLAYER[1]
            x,y = self.player[i]
            ox,oy = self.objective[i]
            world[x,y] = PLAYER[0]
            world[ox,oy] = PLAYER[0]
            observation.append(world)
        return observation

    def step(self,action):
        reward = []
        # First everybody make a move, and check if crash into walls or our of bound
        for i in range(len(action)):
            r = STEP
            x,y = self.player[i]
            move = action[i]
            if self.player[i] == self.objective[i]:
                move = NOP
            if move == UP:
                x-=1
            elif move == DOWN:
                x+=1
            elif move == LEFT:
                y-=1
            elif move == RIGHT:
                y+=1
            if x<0 or x>=self.size or y<0 or y>=self.size or self.world[x,y]!=PATH:
                r+=CRASH
                self.player[i] = (self.size-1,i*self.dist)
            else:
                self.player[i] = (x,y)
            reward.append(r)

        # Now check for crash,
        crash = [False for i in range(len(action))]
        done = [False for i in range(len(action))]
        for i in range(len(action)):
            if self.player[i]== self.objective[i]: # safe place
                done[i] = True
                reward[i]+=REACH_OBJ
                continue
            for j in range(len(action)):
                if i==j:
                    continue
                if self.player[i]==self.player[j]: # sounds like i crashed into j
                    crash[i] = True
                    reward[i]+=CRASH
                    break # you crash with one or multiple is still a crash

        # Once you know the crash status, reset the player who need
        for i in range(len(crash)):
            if crash[i]:
                self.player[i] = (self.size-1,i*self.dist)

        return self.get_observation(),reward,all(done),{}


if __name__ == '__main__':
    import time
    agent_num = 4
    size = 4
    world = World(world_size=size,agent_num=agent_num)
    world.render()
    plt.savefig('intersection_dqn_more_agent/world_4_agents.png')
    time.sleep(10)
    # for i in range(10):
    #     world.add_random_obstacle()
    #     world.render()
        # plt.savefig('intersection_dqn_obstacle/world_demo_{}.png'.format(i))
    # plt.savefig('intersection/world.png')
    # state = world.reset()
    # for i in range(agent_num):
    #     plt.imshow(state[i],cmap=cm,norm=norm)
    #     plt.savefig('intersection/demo_observation_{}.png'.format(i))
    #     plt.show()





