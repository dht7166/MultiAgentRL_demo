import os
os.makedirs('intersection/',exist_ok=True)
import pickle
from tqdm import tqdm
from intersection import *
import numpy as np

class Agent():
    def __init__(self,Q= {},alpha = 0.1,gamma = 0.9):
        self.Q = Q
        self.gamma = gamma
        self.alpha = alpha

    def learn(self,state,action,next_state,reward,done):
        v = self.Q.get((state.tobytes(), action), np.random.random())
        nv = -9999999
        if not done:
            for i in range(5):
                nv = max(nv, self.Q.get((next_state.tobytes(), i), np.random.random()))
        else:
            nv = 0
        self.Q[(state.tobytes(), action)] = v + self.alpha * (reward + self.gamma * nv - v)

    def get_action(self,state,epsilon):
        if np.random.random() <= epsilon:
            return np.random.randint(5)  # 0-4
        best = -999999
        action = 0
        for i in range(5):
            res = self.Q.get((state.tobytes(), i), np.random.random())
            if best < res:
                action = i
                best = res
        return action

    def save(self,name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(self.Q, f, pickle.HIGHEST_PROTOCOL)

    def load(self,name):
        def internal_load(name):
            with open(name + '.pkl', 'rb') as f:
                return pickle.load(f)
        self.Q = internal_load(name)



if __name__ == '__main__':
    agent_num = 3
    size = 5
    world = World(world_size=5,agent_num=agent_num)
    agent = [Agent(Q = {},alpha = 0.1,gamma = 0.9) for i in range(agent_num)]


    # Training params
    epsilon = 0.1  # exploration rate
    n_epochs = 1000000

    reward_this_epoch = 0
    reward_per_epoch = []
    step_this_ep = 0
    step_per_epoch = []
    state = world.reset()
    for episode in tqdm(range(n_epochs)):
        action = [agent[i].get_action(state[i],epsilon) for i in range(agent_num)]
        next_state,reward,done,info = world.step(action)

        # training
        for i in range(agent_num):
            agent[i].learn(state[i],action[i],next_state[i],reward[i],done)

        state = next_state

        # record logistic
        reward_this_epoch+=sum(reward)
        step_this_ep+=1

        if done or step_this_ep>=500:
            reward_per_epoch.append(reward_this_epoch)
            step_per_epoch.append(step_this_ep)
            step_this_ep = 0
            reward_this_epoch = 0
            state = world.reset()


     # save training
    for i in range(agent_num):
        agent[i].save('intersection/save_{}'.format(i))

    # report results
    plt.figure(figsize=(10,10))
    plt.plot(reward_per_epoch)
    plt.savefig('intersection/QL_reward_ep.png')
    plt.show()
    plt.plot(step_per_epoch)
    plt.savefig('intersection/QL_step_to_solve.png')
    plt.show()


