"""
DQN to decide the fraction of power of a battery to charge (or discharge) to earn money in the imbalance settlement.

The training environment does not consider the SOC and the roundtrip efficiency whereas the test environment does.

The program allows to save the results of the best policy at the end.
/!\ The policy is saved but the name depends only on the step number => if the program is run again in the same
    conditions, the policy will be overwritten. If you wanna keep the policy for a longer time, you may wanna change the
    name.

It is implemented using tensor flow agent.

J'ai juste utilisé l'environnement de test pour faire l'entrainement  
"""

from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.policies import PolicySaver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime
from collections import deque

#%%
# ------------------ENVIRONMENT--------------------
class Environment(py_environment.PyEnvironment):

    def __init__(self, observations, non_observable_states, scaler, scalers, capacity: float, discount_rate: float,
                 nb_quarters_per_episode: int, col_MDP: int, num_action: int):

        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype='int32', minimum=0, maximum=num_action-1,
                                                        name="action")
        # Actually, since it has been scaled, the observation should be between -1 and 1 but if you put those bounds,
        # you may encounter some inexplicable errors. Therefore, I put -1.1 and 1.1 arbitrarily to loosen the bounds
        # a little bit
        self._observation_spec = array_spec.BoundedArraySpec(shape=(np.shape(observations)[1],), dtype='float64',
                                                             minimum=-1.1, maximum=1.1, name="observation")
        self._observation_samples = observations
        self._non_observable_samples = non_observable_states  # Basically, this is the real system imbalance
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._observation_index = 0
        self._observation = observations[self._observation_index]
        self._episode_ended = False
        self._capacity = capacity
        self._discount = discount_rate
        self.data_save = deque(maxlen=np.shape(observations)[0])
        self._col_MDP = col_MDP
        self._episode = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False
        return ts.restart(self._observation)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()
        
        #on centre l'action entre -10 et 10 et on normalise entre -1 et 1
        battery_charge_fraction = (action - floor((self._action_spec.maximum - self._action_spec.minimum + 1) / 2)) \
                                  / floor((self._action_spec.maximum - self._action_spec.minimum + 1) / 2)
                                  
        #capacity [MW] * proportion = puissance réelle qu'on va appliquer pdt 15'                           
        battery_charge = battery_charge_fraction * self._capacity
        
        #on retire le scaling sur l'observation du SI
        syst_imb = self._scaler_si.inverse_transform(self._non_observable_samples[self._observation_index].reshape(1, 1))
        real_si = float(syst_imb) - battery_charge

        index = (-real_si / 100) + 6
        index = np.clip(index, 0, 12)
        if index < 6:
            index = floor(index)
        else:
            index = ceil(index)
        MP_price = float(self._scalers_MP[index].inverse_transform(
            np.array(self._observation[self._col_MDP + index], ndmin=2)))
        MP_0_price = float(self._scalers_MP[6].inverse_transform(
            np.array(self._observation[self._col_MDP + 6], ndmin=2)))

        profit = MP_price*(-battery_charge / 4)
        delta_price = MP_price - MP_0_price
        
        #(euro/MWhs)
        reward = float(delta_price) * (-battery_charge / 4)

        self.data_save.append([float(syst_imb), float(action), delta_price, reward, battery_charge, real_si, profit,
                               self._observation_index])
        self._observation_index += 1

        # We are NOT at the end of the dataset
        if self._observation_index < len(self._observation_samples):
            # We are IN an episode
            if self._observation_index < (self._nb_quarters_per_episode*self._episode):
                self._observation = self._observation_samples[self._observation_index]
                return ts.transition(self._observation, reward, discount=self._discount)
            # The episode must end but we are not a the end of the dataset
            else:
                self._observation = self._observation_samples[self._observation_index]
                self._episode_ended = True
                return ts.termination(self._observation, reward)
        # We are at the end of the dataset. We must start again at the beginning. This restart stops the episode
        # (even if it is not finished) and start a new one a the beginning of the dataset.
        else:
            self._observation_index = 0
            self._episode = 0
            self._observation = self._observation_samples[self._observation_index]
            self._episode_ended = True
            return ts.termination(self._observation, reward)


class Environment_test(py_environment.PyEnvironment):

    def __init__(self, observations, non_observable_states, scaler, scalers, max_power: float, discount_rate: float,
                 nb_quarters_per_episode: int, col_MDP: int, EP: float, eta: float, num_action: int):

        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype='int32', minimum=0, maximum=num_action-1,
                                                        name="action")
        # Actually, since it has been scaled, the observation should be between -1 and 1 but if you put those bounds,
        # you may encounter some inexplicable errors. Therefore, I put -1.1 and 1.1 arbitrarily to loosen the bounds
        # a little bit
        self._observation_spec = array_spec.BoundedArraySpec(shape=(np.shape(observations)[1],), dtype='float64',
                                                             minimum=-1.1, maximum=1.1, name="observation")
        self._observation_samples = observations
        self._non_observable_samples = non_observable_states  # Basically, this is the real system imbalance
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._observation_index = 0
        self._observation = observations[self._observation_index]
        self._episode_ended = False
        self._max_power = max_power
        self._discount = discount_rate
        self.data_save = deque(maxlen=np.shape(observations)[0])
        self._col_MDP = col_MDP
        self._episode = 0
        self._EP_ratio = EP
        self._eta = eta

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        #on centre l'action entre -1 et 1 => ça donne  la fraction de puissance max à laquelle on travaille pdt 15'
        charge_fraction_of_max_power = ((action - floor((self._action_spec.maximum - self._action_spec.minimum + 1) / 2))
                                       / floor((self._action_spec.maximum - self._action_spec.minimum + 1) / 2))

        #on met à jour le SoC et on le fixe entre 0 et 1 de sorte que si le soc est hors de cet intervalle
        #on le fixe à 0 ou 1 en fonction de où est la valeur en dehors de l'intervalle
        new_soc = self._observation[-1] + (charge_fraction_of_max_power / (4 * self._EP_ratio))
        new_soc = np.clip(new_soc, 0, 1)
        
        #fraction de la puissance à laquelle on va travailler adaptée par rapport au soc car si le soc est le même qu'avant
        #ça veut dire qu'on peut rien faire et donc l'action charge_fraction_of_max_power -> 0.
        charge_fraction_of_max_power = (new_soc - self._observation[-1]) * (4 * self._EP_ratio)


        #Ce qu'on a fait jusqu'ici = on prend pas en compte le soc et on regarde l'action qu'on aurait fait grâce à quoi
        #on calcule le soc. Ensuite grâce à ce soc et à l'ancien on met à jour l'action choisie pour prendre en compte
        #la limite physique imposée par le soc

        battery_charge_MW = charge_fraction_of_max_power * self._max_power
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1))


        #network charge = ce qu'on transmet au réseau mais on doit affecter le rendement de la batterie
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        real_si = float(syst_imb) - network_charge_MW
        index = (-real_si / 100) + 6
        index = np.clip(index, 0, 12)
        if index < 6:
            index = floor(index)
        else:
            index = ceil(index)

        MP_price = float(self._scalers_MP[index].inverse_transform(
            np.array(self._observation[self._col_MDP + index], ndmin=2)))
        MP_0_price = float(self._scalers_MP[6].inverse_transform(
            np.array(self._observation[self._col_MDP + 6], ndmin=2)))

        #profit = prix de l'énergie * l'énergie qu'on renvoie sur le réseau
        profit = MP_price * (-network_charge_MW / 4)
        
        #delta price est la diff de prix avec le DA
        delta_price = MP_price - MP_0_price
        
        #le reward est l'argent qu'on gagne grâce à notre imbalance (euro/MWh)
        reward = float(delta_price) * (-network_charge_MW / 4)

        self.data_save.append([float(syst_imb), float(action), delta_price, reward, battery_charge_MW, real_si, profit,
                               self._observation_index, self._observation[-1], network_charge_MW])
        self._observation_index += 1

        # We are NOT at the end of the dataset
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, -1] = new_soc
            # We are IN an episode
            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                self._observation = self._observation_samples[self._observation_index]
                return ts.transition(self._observation, reward, discount=self._discount)
            # The episode must end but we are not a the end of the dataset
            else:
                self._observation = self._observation_samples[self._observation_index]
                self._episode_ended = True
                return ts.termination(self._observation, reward)
        # We are at the end of the dataset. We must start again at the beginning. This restart stops the episode
        # (even if it is not finished) and start a new one at the beginning of the dataset.
        else:
            self._observation_index = 0
            self._episode = 0
            self._observation = self._observation_samples[self._observation_index]
            self._episode_ended = True
            return ts.termination(self._observation, reward)



# ----------------------FUNCTIONS----------------------
def floor(x):
    return int(np.floor(x))


def ceil(x):
    return int(np.ceil(x))

#%%
# ----------------IMPORT & PREPROCESSING--------------------

def importdata(nb_rows):
    nb_rows_in_excel = 126144  # total nb of rows in the excel file
    nb_rows_to_skip = nb_rows_in_excel - nb_rows
    df = pd.read_excel(r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\stage Pietro Favaro\HandedIn - shared\Data\Frame_inputs_4_2018.xlsx',
                       skipfooter=nb_rows_to_skip, usecols="B,D:AE")
    
    #add a column of 0 for the soc
    df = df.assign(soc=np.zeros(len(df)))
    train_samples = df.to_numpy().astype("float64")
    #last element of the first row = 0.5 (premier soc=0.5 => on commence avec la batterie à la 1/2 de sa charge)
    train_samples[0, -1] = 0.5

    df2 = pd.read_excel(r'C:\Users\Cyril\OneDrive - UMONS\MA2\TFE\stage Pietro Favaro\HandedIn - shared\Data\Frame_inputs_4_test_2019.xlsx', usecols="B,D:AE")
    df2 = df2.assign(soc=np.zeros(len(df2)))
    test_samples = df2.to_numpy().astype("float64")
    test_samples[0, -1] = 0.5
    
    #vertical concatenate of both samples
    samples = np.concatenate([train_samples, test_samples], axis=0)
    
    #number of the column containing the SI
    index_col_si = 0
    
    #all the SI of the excel in non_obs_exact_SI
    non_observable_exact_si = samples[:, index_col_si]
    
    #obs_samp = samp without SI => the SI is considered non_observable, but we have the quantiles of the predicted SI
    observable_samples = np.delete(samples, index_col_si, 1)

    #position des colonnes de prix dans l'excel
    index_first_col_MDP = 15
    index_last_col_MIP = index_first_col_MDP + 13

    #on ne touche pas aux obs_sample et on dit q'uils sont déjà scaled ?????
    scaled_observable_samples = observable_samples
    
    # We scale down the SI prediction data |b| 0 and 1
    scaler_si = MinMaxScaler(feature_range=(-1, 1))  # creation of the object
    # fit_transform takes only the column with the SI prediction
    scaled_si = scaler_si.fit_transform(non_observable_exact_si.reshape(-1, 1))

    scaler_list = []
    #for j in 0,...,10,15,...,(15+13) => permet de ne pas scale les variables calendaires
    #on scale toutes les variables observables et on stock les scalers des MP dans une liste pour 
    #pouvoir ensuite les utiliser pour la tsfo inverse.
    #Toutes les données scaled sont stockées dans scaled_observable_samples
    for j in np.concatenate((np.arange(11), np.arange(index_first_col_MDP, index_last_col_MIP))):
        scaler_MP = MinMaxScaler(feature_range=(0, 1))
        scaled_MP = scaler_MP.fit_transform(observable_samples[:, j].reshape(-1, 1))
        scaled_observable_samples[:, j] = scaled_MP[:, 0]   
        if index_first_col_MDP <= j <= index_last_col_MIP:
            scaler_list.append(scaler_MP)

    #divise en set entrainement et test les données observables et non-observables
    train_observable_samples = scaled_observable_samples[0:nb_rows, :]
    test_observable_samples = scaled_observable_samples[nb_rows:np.shape(samples)[0], :]
    train_non_observable_samples = scaled_si[0:nb_rows, :]
    test_non_observable_samples = scaled_si[nb_rows:np.shape(samples)[0], :]

    return index_col_si, index_first_col_MDP, scaler_si, scaler_list, train_observable_samples, \
           test_observable_samples, train_non_observable_samples, test_non_observable_samples

#%%
# ----------------- PARAMETERS ---------------------
battery_max_power = 150 #MW

#caract de la batterie donnant le rapport entre l'énergie qu'elle peut stocker sur 
#la puissance qu'elle peut débiter pdt un temps donné 
EP_ratio = 1

#rendement de la batterie (entre ce qu'elle charge et ce qu'elle sera capable de décharger)
roundtrip_efficiency = 0.9

#import data
nb_rows_to_read = 126144  # Advice: Make sure 0.1*nb_rows_to_read is a multiple of "self._nb_quarters_per_episode = 96"
index_col_si, index_first_col_MDP, scaler, scaler_list, train_observable_samples, test_observable_samples, \
train_exact_si_samples, test_exact_si_samples = importdata(nb_rows_to_read)

#actions entre 0 et 20 donc 21 possibilités
action_size = 21

#1 épisode dure 24h (=96*15')
nb_quarters_per_episode = 96

#%%
# -----Hyperparameters-----

#100000 expériences dans le buffer
replay_buffer_max_length = 100000

#le batch size est le nombre d'éléments avec lequels on va entraîner le réseau étape par étape.
#si on a 1000 éléments et que le batch est 50, on va passer 50 par 50 les éléments à l'entraînement
#ce qui permet de réduire la mémoire nécéssaire pour entraîner le réseau. Ca a aussi un impact sur 
#l'entrainement car ça peut accélerer la convergence d'avoir un plus petit batch mais en même temps ça peut mener
#à des instabilités car on met bcp à jour les poids
batch_size = 1024  # une batch size de 32 empêche les couches cachées de plus de 105 neurones de converger
# il faut 128 pour faire fonctionner un réseau de 210*210 en couches cachées

#gamma, la valeur des rewards futur par rapport à la reward actuelle dans le calcul de la Q-val
discount_rate = 0.1

#epsylon-greedy strat
exploration_rate = 0.005    

#learning rate utilisé dans l'entrainement du réseau de neurones
learning_rate = 1e-4

# Initial number of rows played which are stored in the replay buffer (experiences are stored)
initial_collect_steps = batch_size * 2
# each iteration we play X rows from train_samples and we store the result in the replay before.
# Then we train ONCE the agent over a batch.
collect_steps_per_iteration = 4     #it's the X

# nb of states in train_samples you will use after having taken some for initializing the replay buffer
#nbr d'itération d'entraînement sachant qu'une it utilise collect_steps_per_it lignes du dataset
num_iterations = int((len(train_observable_samples) - initial_collect_steps)/collect_steps_per_iteration)

log_interval = floor(num_iterations / 20)  # the loss is printed every "log_interval" iterations.

# nb of episodes (= runs through "self._nb_quarters_per_episode = 96" rows of test_samples) to evaluate the nn before
# training on a random policy and during training on the computed policy.

#nbr d'épisodes qui seront utilisés pour l'éval
num_eval_episodes = int(len(test_observable_samples) / nb_quarters_per_episode)
#toutes les eval_interval it on test le nn
eval_interval = floor(num_iterations / 20)  # the nn is tested every "eval_interval" steps

"""
STRUCTURE OF THE CODE
Here is an explication of the impact of the parameter on the algorithm

1) The algorithm applies a random policy over the first "initial_collect_steps" rows of the train_samples dataset. The
    results are stored in the replay buffer for further use.

2)  The agent's policy is then evaluated over "num_eval_episodes" (=epochs) of the dataset test_samples.

3)  "num_iterations" iterations are executed. During one iteration, "collect_steps_per_iteration" steps 
    (= 1 rows of train_samples) are computed and stored in the replay buffer.

"""

print("Preprocessing done \n")

#%%----------MAIN----------
starting_time = time.time()
# ----------------------MAIN--------------------------

#---creation of the 3 envs---(training, test, test)
train_env = Environment_test(train_observable_samples, train_exact_si_samples, scaler, scaler_list,
                        battery_max_power, discount_rate, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, 
                        roundtrip_efficiency,action_size)
# The python environment we created is wrapped into a tensorflow environment
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

# Environment to evaluate our nn every eval_interval steps
eval_env = Environment_test(test_observable_samples, test_exact_si_samples, scaler, scaler_list, battery_max_power,
                       discount_rate, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                       action_size)
tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

# Second evaluation environment, the same than the training one to compare.
eval_env_2 = Environment_test(test_observable_samples, test_exact_si_samples, scaler, scaler_list, battery_max_power,
                       discount_rate, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                       action_size)
tf_eval_env_2 = tf_py_environment.TFPyEnvironment(eval_env_2)

#%% TEST OF THE ENVIRONMENT (not compulsory) -> not the good packages
#It is a good practice to ensure that the environment is well-configured before using it.
'''

utils.validate_py_environment(env, episodes=5)

empty_battery_action = 0
time_step = env.reset()
cumulative_reward = time_step.reward

for _ in range(nb_rows_to_read):
    time_step = env.step(empty_battery_action)
    cumulative_reward += time_step.reward
    if (_+1)%96 == 0:
        print("Reward episode {}/{}: {}".format((_+1)/96, nb_rows_to_read/96, cumulative_reward))
        cumulative_reward = 0
'''

#%%
# AGENT

#dimensions des couches fully-connected
#la première couche fait 29 neurones car on a rajouté le SoC  
fc_layer_params = (np.shape(train_observable_samples)[1], 1160, 1240)

#création d'un tenseur avec les specs des actions
action_tensor_spec = tensor_spec.from_spec(tf_train_env.action_spec())


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units, activation=tf.keras.activations.relu,
                                 kernel_initializer=tf.keras.initializers.VarianceScaling(
                                     scale=2.0, mode='fan_in', distribution='truncated_normal'))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.

#creation de la couche input et des couches cachées
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]

#creation de la couche output
q_values_layer = tf.keras.layers.Dense(action_size, activation=None,
                                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03,
                                                                                              maxval=0.03),
                                       bias_initializer=tf.keras.initializers.Constant(0.))

#assemblage des couches séquentiellement
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(tf_train_env.time_step_spec(), tf_train_env.action_spec(), q_network=q_net,
                           optimizer=optimizer, epsilon_greedy=exploration_rate,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           gamma=discount_rate, train_step_counter=train_step_counter)

agent.initialize()

# POLICIES
#politique que l'agent suis pour evaluer les perf (=la bonne politique)
eval_policy = agent.policy
#politique appliquée pdt le training qui inclu une partie d'exploration
collect_policy = agent.collect_policy

# a random policy to generate the first experiences that will be stored in the replay buffer
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

#%%+
# --------------METRICS&EVALUATION--------------

def compute_avg_return(environment, policy, num_epochs=1):
    total_return = 0.0
    counter = 0
    for _ in range(num_epochs):

        time_step = environment.reset()
        episode_return = 0.0
        
        #si num_epochs=1 ça veut dire qu'on va calculer le reward sur un épisode=24h
        while not time_step.is_last():
            action_step = policy.action(time_step)      #choix de l'action en fct de l'observation de l'état au temps time_step
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            counter += 1
        total_return += episode_return

    avg_return = total_return / counter
    return avg_return.numpy()[0]

#%%
# --------------REPLAY BUFFER--------------
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_train_env.batch_size,
                                                               max_length=replay_buffer_max_length)

#%%
# -------------------DATA COLLECTION-----------------
#fct pour ajouter les expériences (trajectories) dans le buffer
def collect_data(environment, policy, buffer, steps):
    reward = 0
    for _ in range(steps):
        time_step = environment.current_time_step()
        reward += time_step.reward
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)
        if next_time_step.is_last():
            environment.reset()

    return reward

#%% ------remplissage initial du buffer avec la politique random + convertir buffer en dataset (pq ?)------
print("\nDATA COLLECTION \n")
collect_data(tf_train_env, random_policy, replay_buffer, initial_collect_steps)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

#%%
# --------------TRAINING THE AGENT---------------

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
print("\nPOLICY EVALUATION B| TRAINING \n")
avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
print("Average return (before training): ", avg_return)
returns = [avg_return]
train_rewards = []
data_save = []
average_train_rewards = []
saver = PolicySaver(eval_policy)

print("\nTRAINING \n")
for _ in range(num_iterations):

    # Collect a few steps using collect_policy and save to the replay buffer.
    train_rewards.append(collect_data(tf_train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration))

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss   #mise à jour des poids + enregistrement de la loss

    step = agent.train_step_counter.numpy()     #numéro de l'itération/step d'entraînement


    #on print la loss tous les log_interv et on enregistre le reward du training
    if step % log_interval == 0:
        average_train_rewards.append(np.sum(train_rewards) / (log_interval * collect_steps_per_iteration))
        print('step = {0}: loss = {1}\taverage reward = {2}'.format(step, int(train_loss), int(average_train_rewards[-1])))
        train_rewards = []


    #on évalue la politique tous les eval_interval et on enregistre la politique à ce moment de l'entraînement
    if step % eval_interval == 0:
        avg_return = compute_avg_return(tf_eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, int(avg_return)))
        saver.save('policy_%d' % step)
        returns.append(avg_return)
        avg_return = compute_avg_return(tf_eval_env_2, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return (same env. than training) = {1}'.format(step, int(avg_return)))

#%% 
# ---------------------PLOT---------------------
def plotdata(y_data, interval, title):
    iterations = np.arange(len(y_data)) * interval
    plt.plot(iterations, y_data)
    plt.title(title)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    plt.ylim()
    plt.grid()
    plt.show()


plotdata(average_train_rewards, log_interval, "Training")
plotdata(returns, eval_interval, "Test")

#%%
# ---------------------SAVE DATA-------------------
def savetoexcel(data_save, name, n_layers, action_size, activ_fct, batch_size, learning_rate, gamma,
                steps_per_iteration, EP, P_max, eta):
    dic_model_features = {"Layers": np.concatenate([n_layers, np.array(action_size, ndmin=1)]), "Activation fct": activ_fct,
                          "Batch size": batch_size, "Collect steps per iteration": steps_per_iteration,
                          "Learning rate": learning_rate, "Discount rate": gamma, "E/P": EP, "P max (MW)": P_max,
                          "Roundtrip efficiency": eta}
    df_model_features = pd.DataFrame(dic_model_features)

    data_save_array = np.asarray(data_save, dtype=object)
    dic_results = {"Observation index": data_save_array[:, 7],"Initial SI": data_save_array[:, 0],
                   "Actions": data_save_array[:, 1], "SOC (pu)": data_save_array[:, 8],
                   "Battery charges (MW)": data_save_array[:, 4], "Network charge (MW)": data_save_array[:, 9],
                   "Real SI": data_save_array[:, 5], "Delta price": data_save_array[:, 2],
                   "Rewards": data_save_array[:, 3], "Profits": data_save_array[:, 6]}
    df_results = pd.DataFrame(dic_results)

    index_col_reward = 3
    average_reward = np.mean(data_save_array[:, index_col_reward])

    with pd.ExcelWriter(name + "_" + str(int(average_reward)) + "_2019.xlsx") as writer1:
        df_model_features.to_excel(writer1, sheet_name='Model Features', index=False)
        df_results.to_excel(writer1, sheet_name='Results', index=False)


running_time = time.time() - starting_time
print("Running time: {}s".format(running_time))

#-----------------------SAVE BEST POLICY---------------------------
policy_step_number = input("Enter the step number of the policy of which you wish to save the results:")
policy_name = "policy_" + policy_step_number

loaded_policy = tf.compat.v2.saved_model.load(policy_name)
compute_avg_return(tf_eval_env, loaded_policy, int(len(test_observable_samples)/nb_quarters_per_episode))
savetoexcel(eval_env.data_save, policy_name, fc_layer_params, action_size, "relu", batch_size, learning_rate,
            discount_rate, collect_steps_per_iteration, EP_ratio, battery_max_power, roundtrip_efficiency)

