import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# random walk RL problem implementation
# inspired by: https://medium.com/@violante.andre/simple-reinforcement-learning-temporal-difference-learning-e883ea0d65b0

def initialize_values(N_states,state_reward):
    
    values = np.zeros((N_states,))              # initial guess of state values
    values_gt = np.zeros((N_states,))           # ground truth state values
    
    state_reward = N_states - 1                 # reward given in highest state
    for s in range(0,N_states):
        values[s] = round(1 / N_states,2)
        
        # probabilities for each state to gain a reward with respect to ending in the reward state
        N_states_other = N_states - 1                       # number of other possible states
        N_states_toreward = state_reward - s                # number of states separating current state and reward state
        prob = 1 - N_states_toreward / N_states_other       # probability of reward is higher the closer the agent gets
        values_gt[s] = round( prob, 2 )
    
    return values,values_gt

if __name__ == '__main__':

    ##############
    # PARAMETERS #
    
    # environment parameters
    N_states = 7
    state_reward = N_states - 1
    state_start = int(N_states/2)
    
    # episodes parameters
    N_episodes = 200

    # hyperparameters
    gamma = 1.                 # discount rate, none in this case
    alpha = .1                 # learning rate
    
    ##############

    # initialize state values
    values,values_gt = initialize_values(N_states,state_reward)
    
    # perform one random walk starting from state = 0
    epdf = pd.DataFrame()
    for e in range (0,N_episodes):
        
        state = state_start
        
        # end episode if in one of the terminal states
        while (state != state_reward) and (state != 0):
            move = np.random.choice([-1,1])
            
            # take action
            state_new = state + move
            
            # observe reward in the state
            if state_new == state_reward:
                reward = 1.0
            else:
                reward = 0.0
            
            # update value estimate for this timestep by scaling prediction error
            values[state] = values[state] + alpha * ( reward + gamma*values[state_new]-values[state])
            
            state = state_new
        
        epdf[e] = values

epdf.iloc[-1,:] = 0.
epdf.iloc[0,:] = 0.

# visualize the data

fig,(ax,cax) = plt.subplots(1,2,figsize=(7,3),gridspec_kw={'width_ratios':[50,1]})
ax.set_title(r'$\alpha$ = {}'.format(alpha))
im = ax.imshow(epdf,aspect='auto',interpolation='none',vmin=0.,vmax=1.,cmap='inferno')
ax.set_xlabel('Episode')
ax.set_ylabel('State')

ticks = [0,0.5,1]
cbar = fig.colorbar(im, cax=cax,ticks=ticks,orientation='vertical')  
cbar.set_label('Value',va='top',ha='left',rotation=90,in_layout=True)

fig.tight_layout()
plt.show()
    
# insights
# can't update value estimation of the terminal states
# probabilities don't need to sum to 1.0