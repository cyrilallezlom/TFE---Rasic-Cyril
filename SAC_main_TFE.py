# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:03:27 2024

@author: Cyril

SAC
"""

#%%
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network


from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.policies import actor_policy
from tf_agents.policies import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import time

from DDPG_classes_TFE_BattCost import floor, ceil, Environment, importdata, compute_avg_return, collect_data, plotdata, Visualize

from tf_prioritized_replay_buffer import TFPrioritizedReplayBuffer


#%% Fixer les seeds pour la reproductibilité
seed = 4
np.random.seed(seed)
tf.random.set_seed(seed)
#%% -------Parameters + Data-------

#battery param
battery_max_power = 150 #MW

EP_ratio = 1

roundtrip_efficiency = 0.9

battery_replacement_cost = 0


#data path
path_training = r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\RL\Data\DataPietro\Frame_inputs_4_2018.xlsx'
path_validation = r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\RL\Data\DataPietro\Frame_inputs_4_test_2019.xlsx'
path_test = r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\RL\Data\DataPietro\Frame_inputs_4_test_2019.xlsx'


#data import
print('Importing data + preprocessing...')
index_col_si, index_first_col_MDP, scaler, scaler_list, train_observable_samples, validation_observable_samples, test_observable_samples, \
train_exact_si_samples, validation_exact_si_samples, test_exact_si_samples = importdata(path_training, path_validation, path_test)
print('Preproced data imported')

#%% -------Hyperparameters-------

#nb of steps per ep
nb_quarters_per_episode = 96

replay_buffer_max_length = 50000

batch_size = 1024  

#number of epochs in the training
num_epochs = 1

#gamma, discounting rate of futures rewards
gamma = 0.1

#tau, soft update of the targets
tau = 1e-3

#learning rates
actor_lr = 1e-3
critic_lr = 1e-3
alpha_lr = 1e-3

#learning weights : what's important is the relative diff of the weights: 1-1-1 means that in the training optimizing actor, critic and alpha is equivalently important
actor_loss_weight = 1
critic_loss_weight = 1
alpha_loss_weight = 1

#optimizers
actor_optimizer = tf.keras.optimizers.Adam(actor_lr, clipnorm=1)
critic_optimizer = tf.keras.optimizers.Adam(critic_lr, clipnorm=1)
alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)

#policy of the actor
actor_policy = actor_policy.ActorPolicy

actor_update_T = 2
target_update_T = 1

#architecture, hidden layers of the nn
hidden_actor = (150, 150)
hidden_critic = (500, 500)

#noise
noise_stddev = 0.2

#number of experiences initially added in the buffer
initial_collect_steps = batch_size * 2

#number of experiences stored at each training iteration in the buffer
collect_steps_per_iteration = 4

num_iterations = num_epochs * int((len(train_observable_samples) - initial_collect_steps)/collect_steps_per_iteration)

num_training_episodes = int(num_iterations/nb_quarters_per_episode)

log_interval = floor(num_iterations / 20)  # the loss is printed every "log_interval" iterations.

eval_interval = floor(num_iterations / 20)  # the agent is tested every "eval_interval" steps

num_test_episodes = int(len(test_observable_samples) / nb_quarters_per_episode)

num_validation_episodes = int(len(validation_observable_samples) / nb_quarters_per_episode)


#%% -------Environments-------
starting_time = time.time()

#Training env 
train_env = Environment(train_observable_samples, train_exact_si_samples, scaler, scaler_list, battery_max_power,
                       gamma, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency, battery_replacement_cost)

#The python environment we created is wrapped into a tensorflow environment
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)


#Validation env 
validation_env = Environment(validation_observable_samples, validation_exact_si_samples, scaler, scaler_list, battery_max_power,
                       gamma, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency, battery_replacement_cost)

#The python environment we created is wrapped into a tensorflow environment
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)


#Testing env 
test_env = Environment(test_observable_samples, test_exact_si_samples, scaler, scaler_list, battery_max_power,
                       gamma, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency, battery_replacement_cost)

#The python environment we created is wrapped into a tensorflow environment
tf_test_env = tf_py_environment.TFPyEnvironment(test_env)


#%% -------Actor and critic-------
Actor = actor_network.ActorNetwork(tf_train_env.observation_spec(),tf_train_env.action_spec(),hidden_actor)
Critic = critic_network.CriticNetwork((tf_train_env.observation_spec(),tf_train_env.action_spec()), joint_fc_layer_params=hidden_critic)


#%% -------Agent-------
train_step_counter = tf.Variable(0)

agent = sac_agent.SacAgent(tf_train_env.time_step_spec(), tf_train_env.action_spec(), Critic, Actor, alpha_optimizer = alpha_optimizer,
                           actor_optimizer=actor_optimizer, critic_optimizer = critic_optimizer, actor_loss_weight=actor_loss_weight,
                           critic_loss_weight=critic_loss_weight, alpha_loss_weight=alpha_loss_weight, actor_policy_ctor=actor_policy,
                           target_update_tau = tau, target_update_period = target_update_T, td_errors_loss_fn = tf.math.squared_difference, gamma = gamma,  
                           reward_scale_factor = 1.0, gradient_clipping = None, debug_summaries=None, summarize_grads_and_vars = False, 
                           train_step_counter=train_step_counter)

agent.initialize()


# -------policies-------
#the real policy of the agent 
eval_policy = agent.policy

#policy including noise
collect_policy = agent.collect_policy

# a random policy to generate the first experiences that will be stored in the replay buffer
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())


#%% -------Replay buffer-------

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_train_env.batch_size,
                                                               max_length=replay_buffer_max_length)
'''
replay_buffer = TFPrioritizedReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_train_env.batch_size,
    max_length=replay_buffer_max_length)
    

beta_PER_fn = tf.keras.optimizers.schedules.PolynomialDecay(
  initial_learning_rate=0.00,
  end_learning_rate=1.00,
  decay_steps = num_iterations)
'''
#%% -------Initial fill-------
print("\nData collection ... \n")
collect_data(tf_train_env, random_policy, replay_buffer, initial_collect_steps)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).shuffle(buffer_size=replay_buffer_max_length).prefetch(3)


#smart iterator to train on all the experiences
iterator = iter(dataset)


#%% -------Training-------
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
print("\nPOLICY EVALUATION B| TRAINING \n")
avg_return = compute_avg_return(tf_test_env, agent.policy, num_test_episodes)
print("Average return (before training): ", avg_return)


returns = [avg_return]
train_rewards = []
data_save = []
average_train_rewards = []
train_loss_plot_critic = []
train_loss_plot_actor = []
saver = PolicySaver(eval_policy)

i = 0

print("\nTRAINING \n")
for _ in range(num_training_episodes):
    
    for _ in range(nb_quarters_per_episode):
        
        # Collect a few steps using collect_policy and save to the replay buffer.
        train_rewards.append(collect_data(tf_train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration))

        # Sample a batch of data from the buffer and update the agent's networks.
        experience, unused_info = next(iterator)

        
        train_loss = agent.train(experience).loss
        
        
        #number of the training iteration
        step = agent.train_step_counter.numpy()    
        
        i+=1

        #on print la loss tous les log_interv et on enregistre le reward du training
        if step % log_interval == 0:
            average_train_rewards.append(np.sum(train_rewards) / (log_interval * collect_steps_per_iteration))
            print('step = {0}: loss = {1}\taverage reward = {2}'.format(step, int(train_loss), int(average_train_rewards[-1])))
            train_rewards = []


        #on évalue la politique tous les eval_interval et on enregistre la politique à ce moment de l'entraînement
        if step % eval_interval == 0:
            avg_return = compute_avg_return(tf_validation_env, agent.policy, num_validation_episodes)
            print('step = {0}: Average Return = {1}'.format(step, int(avg_return)))
            returns.append(avg_return)


#%% -------plot-------
plotdata(average_train_rewards, log_interval, "Training")
plotdata(returns, eval_interval, "Validation")
plt.plot(train_loss_plot)

#%% -------Test of the agent-------
#avg_test_return = compute_avg_return(tf_test_env, agent.policy, num_test_episodes)
avg_test_return, ac = compute_avg_return(tf_test_env, agent.policy, num_test_episodes)

print('testing average return (per step)',avg_test_return)
#%% -------Saving-------
path = r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\RL\DDPG\enregistrementAgent\Agent1'
#%%
# Enregistrez le modèle
tf.saved_model.save(agent, path)

#%% -------Loading-------
agentDDPG = tf.saved_model.load(path)


#%% -------Visualize-------
actions, rewards = Visualize(tf_test_env, agent.policy, 48)


col1 = 'actions'
col2 = 'rewards'
data = pd.DataFrame({col1: actions, col2: rewards})
data.to_excel("comparaison.xlsx", sheet_name="DDPG", index=False)