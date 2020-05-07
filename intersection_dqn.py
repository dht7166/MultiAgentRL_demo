import os
folder = 'intersection_dqn_333'
os.makedirs(folder,exist_ok=True)

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense,ReLU

from collections import deque
import numpy as np
import random
from intersection import *
from matplotlib import pyplot as plt
plt.figure(figsize=(8,8))

from tqdm import tqdm

class Buffer:
    def __init__(self,size = 5000):
        self.buffer = deque(maxlen=size)

    def push(self,state,action,next_state,reward,done):
        if done:
            next_state = None
        else:
            next_state = next_state.flatten()
        self.buffer.append((state.flatten(),action,next_state,reward))

    def sample(self,batch_size):
        return random.sample(self.buffer,batch_size)

class Agent:
    def __init__(self,input_dim,output_dim,batch_size = 32,gamma = 0.9):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = self.make_model(lr = 0.01)
        self.target_model = self.make_model()
        self.target_model._make_predict_function()
        self.model._make_predict_function()


    def make_model(self,lr = 0.01):
        inp = Input((self.input_dim,))
        x = Dense(128)(inp) # prev 128
        x = ReLU()(x)

        x = Dense(128)(x)
        x = ReLU()(x)

        out = Dense(self.output_dim)(x)

        model = Model(inp,out)
        opt = Adam(lr = lr)
        model.compile(opt,loss='huber_loss')
        return model

    def copy_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self,experience):
        X = np.zeros((self.batch_size, self.input_dim))
        Y = np.zeros((self.batch_size, self.output_dim))
        for i in range(self.batch_size):
            state, action, next_state, reward = experience[i]
            X[i] = state
            target = self.model.predict(state[np.newaxis, :])[0]
            if next_state is None:
                target[action] = reward
            else:
                next_target = np.amax(self.target_model.predict(next_state[np.newaxis,:]))
                target[action] = reward + self.gamma*next_target
            Y[i] = target
        self.model.fit(X,Y,batch_size=self.batch_size, verbose= 0)

    def get_action(self,state,epsilon = 0.05):
        if np.random.random()<epsilon:
            return np.random.randint(0,self.output_dim)
        return np.argmax(self.model.predict(state.flatten()[np.newaxis,:])[0])

if __name__ == '__main__':
    # Initialize agent and environment

    """ Previous setting that worked for normal dqn
    agent_num = 3 
    size = 5
    """
    """ Previous setting that worked for normal dqn with obstacle
        agent_num = 2
        size = 5
    """
    """
    More_agent 4 - 4
    """

    agent_num = 3
    size = 3 # usual size is 5
    world = World(world_size=size,agent_num=agent_num)
    agent = [Agent(size*size,5) for i in range(agent_num)]
    buffer = [Buffer(20000) for i in range(agent_num)]

    # Load old files
    for i in range(agent_num):
        try:
            agent[i].model.load_weights(folder+'/save_{}.h5'.format(i))
            agent[i].target_model.load_weights(folder + '/save_{}.h5'.format(i))
        except:
            pass


    # Training specifics setting
    add_obstacle = False
    batch_size = 32
    n_epochs = 100000 # even 10000 works ok for normal dqn
    n_update_freq = 2000
    n_report_freq = 10000
    n_step_per_ep = 100
    n_epoch_per_change = 10
    initial_epsilon = 1
    final_epsilon = 0.01
    epsilon_decay = (initial_epsilon-final_epsilon)/(n_epochs*0.75)
    epsilon_fn = lambda step : max(initial_epsilon-epsilon_decay*step,final_epsilon)

    # start training
    reward_this_epoch = 0
    reward_per_epoch = []
    step_this_ep = 0
    step_per_epoch = []
    state = world.reset()
    for step in tqdm(range(n_epochs)):
        eps = epsilon_fn(step)
        action = [agent[i].get_action(state[i],eps) for i in range(agent_num)]
        next_state,reward,done,info = world.step(action)

        for i in range(agent_num): # push 2 buffer
            buffer[i].push(state[i],action[i],next_state[i],reward[i],done)
        state = next_state

        # logistic stuff, keeping results n shit
        reward_this_epoch+= sum(reward)
        step_this_ep+=1
        if done or step_this_ep>=n_step_per_ep:
            reward_per_epoch.append(reward_this_epoch)
            step_per_epoch.append(step_this_ep)
            reward_this_epoch = 0
            step_this_ep = 0
            if len(step_per_epoch)%n_epoch_per_change == 0 and add_obstacle:
                world.add_random_obstacle()
            state = world.reset()
        if len(buffer[0].buffer)<batch_size: # training only if enough to get train
            continue

        # Training yay
        for i in range(agent_num):
            experience = buffer[i].sample(batch_size)
            agent[i].train(experience)

        if (step+1)%n_update_freq == 0:
            for i in range(agent_num):
                agent[i].copy_weights()
                agent[i].model.save_weights(folder+'/save_{}.h5'.format(i))
        if (step+1)%n_report_freq == 0:
            print("STEP {}, EP {}, AVE REWARD {}, AVE COMPL STEP {}, CURRENT EPS {}".format(step+1,len(reward_per_epoch),
                                                                                            sum(reward_per_epoch[-10:])/10,
                                                                                            sum(step_per_epoch[-10:])/10,
                                                                                            eps))
            plt.figure(figsize=(8,8))
            plt.plot(reward_per_epoch)
            plt.title("EP {} STEP {}".format(len(reward_per_epoch),step))
            plt.savefig(folder+'/DQN_reward_ep.png')
            plt.clf()

            plt.plot(step_per_epoch)
            plt.title("EP {} STEP {}".format(len(reward_per_epoch), step))
            plt.savefig(folder+'/DQN_step_to_done.png')