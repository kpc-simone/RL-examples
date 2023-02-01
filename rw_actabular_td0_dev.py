import numpy as np

class ACAgent:
    def __init__(self,
            n_states_ = 50,   
            actions_ = [-1,0,1],     # move left, stay in place, move right
            state_start_ = 25,
            alpha_ = .1,             # learning rate
            gamma_ = 1.,             # discount rate, in this case none
            epsilon_ = 0.75          # probability of moving randomly
            ):
        
        self.n_states = n_states_
        self.actions = actions_
        self.alpha = alpha_
        self.gamma = gamma_
        self.epsilon = epsilon_
        self.choice = 0
        self.values = np.random.uniform( low = 0., high = 1., size = self.n_states ) / 1000
        self.policy = np.random.uniform( low = 0., high = 1., size = (self.n_states,len(self.actions)) ) / 1000
        
    ### CRITIC ###
    def compute_td_error(self,state,next_state,reward):
        # TD error is reward prediction error
        self.td_error = reward + self.gamma * self.values[next_state] - self.values[state]
        
    def update_value(self,state,next_state,reward):
        # update value estimate for this timestep by scaling TD error by learning rate
        # if td_error is positive, increases value of state
        # if td_error is negative, decreases value of state
 
        self.compute_td_error(state,next_state,reward)
        self.values[state] = self.values[state] + self.alpha * self.td_error
    
    ### ACTOR ###    
    def select_action(self,state):
        # select action to exploit (move in direction of highest predicted value, i.e. greedily)
        # or explore (i.e. randomly)
        
        p = np.random.uniform( low = 0., high = 1.)
        if p > self.epsilon:
            self.choice = np.argmax(self.policy[state,:])
            action = self.actions[ self.choice ]
        else:
            self.choice = np.random.randint(0,len(self.actions))
            action = self.actions[self.choice]
        return action
        
    def update_policy(self,state):
        self.policy[state,self.choice] = self.policy[state,self.choice] + self.alpha * self.td_error
    
class Environment:
    def __init__(self,
            n_states_ = 100,
            states_reward_ = [10,50],
            state_start_ = 25,
            max_steps_ = 100
            ):
            
        self.n_states = n_states_
        self.reward_states = states_reward_
        
        self.state_start = np.random.randint(low = 0, high = self.n_states)
        self.state = self.state_start
        self.next_state = self.state_start
        self.reward_counter = 0
        self.step_count = 0
        self.max_steps = max_steps_
            
    def reset(self):
        self.state_start = np.random.randint(low = 0, high = self.n_states-1)
        self.state = self.state_start
        self.next_state = self.state_start
        self.done = False
        self.reward = 0.
        self.reward_counter = 0
        self.step_count = 0
    
    def step(self,action):
        self.step_count += 1

        self.next_state = self.state + action
        
        if self.next_state == 0 or self.next_state == self.n_states-1:
            self.done = True
        elif self.reward_counter == 1:
            self.done = True
        elif self.next_state in self.reward_states:
            self.reward = 1.0 / ( 0.5 * self.step_count )
            self.reward_counter += 1
        elif self.step_count > self.max_steps:
            self.done = True
        
        return self.next_state,self.reward,self.done

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    ### PARAMETERS ###
    # script parameters
    n_episodes = 2500
    n_states = 100
    actions = [-1,0,1]
    
    # agent hyperparameters
    gamma = 1.                 # discount rate, none in this case
    alpha = .1                 # learning rate
    epsilon = .75
    
    # environment parameters
    state_start = int(n_states/2)
    states_reward = [33,67]
    ##################


    # create and initialize the agent
    agent = ACAgent(n_states_ = n_states,
                        gamma_ = gamma,
                        alpha_ = alpha,
                        epsilon_ = epsilon)
    environment = Environment(n_states_ = n_states, 
                                states_reward_ = states_reward, 
                                state_start_ = state_start)
    
    # structure for saving the data
    vdf = pd.DataFrame()                    # value function estimates per episode
    pdf = pd.DataFrame()
    
    for e in range(0,n_episodes):
        print(e)
        environment.reset()
        done = False
        while not done:
            action = agent.select_action(environment.state)
            next_state,reward,done = environment.step(action)
            
            agent.update_value(environment.state,environment.next_state,reward)
            agent.update_policy(environment.state)
            environment.state = environment.next_state
            
        vdf[e] = agent.values
        pdf[e] = [ agent.actions[i] for i in np.argmax(agent.policy,axis=1)]
            
    vdf.iloc[0,:] = 0.
    vdf.iloc[0,:] = 0.

    # visualize the data

    fig,axes = plt.subplots(2,2,figsize=(7,7),gridspec_kw={'width_ratios':[50,1]})
    
    # plot learning of value function
    ax = axes[0,0]
    cax = axes[0,1]
    ax.set_title(r'$\alpha$ = {}'.format(alpha))
    im = ax.imshow(vdf,aspect='auto',interpolation='none',vmin=0.,vmax=10.,cmap='inferno')
    ax.set_xlabel('Episode')
    ax.set_ylabel('State')
    ax.set_ylim(0,100)
    ticks = [0,5.,10.]
    cbar = fig.colorbar(im, cax = cax, ticks = ticks, orientation='vertical')  
    cbar.set_label('Value',va='top',ha='left',rotation=90,in_layout=True)
    
    for state_reward in states_reward:
        ax.axhline(state_reward,color='dimgray',linestyle='--')
    
    # plot learning of policy
    ax = axes[1,0]
    cax = axes[1,1]
    im = ax.imshow(pdf,aspect='auto',interpolation='none',vmin=-1.,vmax=1.,cmap='viridis')
    ax.set_xlabel('Episode')
    ax.set_ylabel('State')
    ax.set_ylim(0,100)
    ticks = [-1,0,1]
    ticklabels = [r'$\downarrow$',r'$\times$',r'$\uparrow$']
    cbar = fig.colorbar(im, cax = cax, ticks = ticks, orientation='vertical')  
    cbar.set_label('Policy',va='top',ha='left',rotation=90,in_layout=True)  
    cbar.ax.set_yticklabels(ticklabels)
    
    for state_reward in states_reward:
        ax.axhline(state_reward,color='white',linestyle='--')

    fig.tight_layout()
    plt.show()        