# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:11:49 2024

@author: Cyril Rasic

Classes and usefull function for DDPG_Main considering cost of battery cycles
"""
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np


from tf_agents.environments import py_environment
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import deque

#%% -------Environment class-------
class Environment(py_environment.PyEnvironment):
    
    #self est l'instance (l'objet de la classe qu'on a instancié = créé)
    def __init__(self, observations, non_observable_states, scaler, scalers, max_power: float, discount_rate: float,
                 nb_quarters_per_episode: int, col_MDP: int, EP: float, eta: float):
    
    
        #super() appelle le constructeur de la classe parente, ça permet d'initialiser les 
        #membres de la classe parente dans cette classe
        super().__init__()
        
        #definition des specs des actions qui peuvent être prises : float entre -1 et 1
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype='float64', minimum=-1.0, maximum=1.0,
                                                        name="action")
        
        # Actually, since it has been scaled, the observation should be between -1 and 1 but if you put those bounds,
        # you may encounter some inexplicable errors. Therefore, I put -1.1 and 1.1 arbitrarily to loosen the bounds
        # a little bit
        self._observation_spec = array_spec.BoundedArraySpec(shape=(np.shape(observations)[1],), dtype='float64',
                                                             minimum=-1.1, maximum=1.1, name="observation")
        #ce qui est observable : les entrées 
        self._observation_samples = observations
        
        #ce qui n'est pas observable : le SI
        self._non_observable_samples = non_observable_states 
        
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
        return ts.restart(self._observation)    #réinitialise l'état initial à observation, laquelle a été mise à jour
                                                #à l'état suivant dans la méthode step.

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        #La fraction de Pmax (entre -1 et 1) à laquelle la batterie va se charger pdt 15'
        fraction_of_max_power = action
        
        #on met à jour le SoC et on le fixe entre 0 et 1 (fct np.clip) de sorte que si le soc est hors de cet intervalle
        #on le fixe à 0 ou 1 en fonction de où est la valeur en dehors de l'intervalle
        new_soc = self._observation[-1] + (fraction_of_max_power / (4 * self._EP_ratio))
        
        
        new_soc = np.clip(new_soc, 0, 1)
        #fraction de la puissance à laquelle on va travailler adaptée par rapport au soc car si le soc est le même qu'avant
        #ça veut dire qu'on peut rien faire et donc l'action charge_fraction_of_max_power -> 0.
        fraction_of_max_power = (new_soc - self._observation[-1]) * (4 * self._EP_ratio)
    
        #Ce qu'on a fait jusqu'ici = on prend pas en compte le soc et on regarde l'action qu'on aurait fait grâce à quoi
        #on calcule le soc. Ensuite grâce à ce soc et à l'ancien on met à jour l'action choisie pour prendre en compte
        #la limite physique imposée par le soc

        #La vraie puissance en MW de la batterie pdt 15'
        battery_charge_MW = fraction_of_max_power * self._max_power
        
        #on utilise le vrai SI observé après coup pour calculer le reward qu'on a gagné
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1))


        #network charge = ce qu'on prend au réseau mais on doit affecter le rendement de la batterie et on
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)
            
        

        #le vrai SI est corrigé par ce qu'on a pris nous au réseau
        real_si = float(syst_imb) - network_charge_MW
        
        #on va rechercher dans les data le prix associé par SI entre -600 et +600MW
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
        
        #delta price est la diff de prix avec le cas pas d'imballance
        delta_price = MP_price - MP_0_price
        
        #le reward est l'argent qu'on gagne grâce à notre imbalance (euro/MWh)
        reward = float(delta_price) * (-network_charge_MW / 4)

        self.data_save.append([float(syst_imb), float(action), delta_price, reward, battery_charge_MW, real_si, profit,
                               self._observation_index, self._observation[-1], network_charge_MW])
        self._observation_index += 1

        # We are NOT at the end of the dataset
        if self._observation_index < len(self._observation_samples):
            
            #on met à jour le soc dans le nouvel état observé
            self._observation_samples[self._observation_index, -1] = new_soc
            
            # We are IN an episode
            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                #update the observed state by the next state
                self._observation = self._observation_samples[self._observation_index]
                #ça retourne une transition constituée de s_t, a_t et s_t+1
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
#%% -------Usefull func-------
def floor(x):
    return int(np.floor(x))


def ceil(x):
    return int(np.ceil(x))

def plotdata(y_data, interval, title):
    iterations = np.arange(len(y_data)) * interval
    plt.plot(iterations, y_data)
    plt.title(title)
    plt.ylabel('Average Return (per steps)')
    plt.xlabel('Iterations')
    plt.ylim()
    plt.grid()
    plt.show()


#%% -------Import data and preprocessing-------
def importdata(path_training, path_validation, path_test):
    
    df = pd.read_excel(path_training, usecols="B,D:AE")
    
    #add a column of 0 for the soc
    df = df.assign(soc=np.zeros(len(df)))
    
    train_samples = df.to_numpy().astype("float64")
    
    #last element of the first row = 0.5 (first soc=0.5 => start the 1st ep with a half charged battery)
    train_samples[0, -1] = 0.5

    df2 = pd.read_excel(path_validation, usecols="B,D:AE")
    df2 = df2.assign(soc=np.zeros(len(df2)))
    validation_samples = df2.to_numpy().astype("float64")
    validation_samples[0, -1] = 0.5
    
    df3 = pd.read_excel(path_test, usecols="B,D:AE")
    df3 = df3.assign(soc=np.zeros(len(df3)))
    test_samples = df3.to_numpy().astype("float64")
    test_samples[0, -1] = 0.5
    
    nb_rows_train = len(df)
    nb_rows_validation = len(df2)
    nb_rows_test = len(df3)
    
    
    #vertical concatenate of the samples
    samples = np.concatenate([train_samples, validation_samples, test_samples], axis=0)
    
    #number of the column containing the SI
    index_col_si = 0
    
    #all the SI of the excel in non_obs_exact_SI
    non_observable_exact_si = samples[:, index_col_si]
    
    #obs_samp = samp without SI => the SI is considered non_observable, but we have the quantiles of the predicted SI
    observable_samples = np.delete(samples, index_col_si, 1)

    #numbers of the prices column
    index_first_col_MDP = 15
    index_last_col_MIP = index_first_col_MDP + 13

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
    train_observable_samples = scaled_observable_samples[0:nb_rows_train, :]
    validation_observable_samples = scaled_observable_samples[nb_rows_train:nb_rows_train+nb_rows_validation, :]
    test_observable_samples = scaled_observable_samples[nb_rows_train+nb_rows_validation:nb_rows_train+nb_rows_validation+nb_rows_test, :]
    
    train_non_observable_samples = scaled_si[0:nb_rows_train, :]
    validation_non_observable_samples = scaled_si[nb_rows_train:nb_rows_train+nb_rows_validation, :]
    test_non_observable_samples = scaled_si[nb_rows_train+nb_rows_validation:nb_rows_train+nb_rows_validation+nb_rows_test, :]

    return index_col_si, index_first_col_MDP, scaler_si, scaler_list, train_observable_samples, validation_observable_samples, \
           test_observable_samples, train_non_observable_samples, validation_non_observable_samples, test_non_observable_samples

#%% -------Metrics and evaluation-------
def compute_avg_return(environment, policy, nbr_episodes):
    
    total_return = 0.0
    counter = 0
    
    for _ in range(nbr_episodes):

        time_step = environment.reset()
        episode_return = 0.0
        
        while not time_step.is_last():
            action_step = policy.action(time_step)    
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            counter += 1
        total_return += episode_return

    avg_return = total_return / counter
    return avg_return.numpy()[0]


#%% -------Test to save actions and reward at each step-------
def Visualize(environment, policy, nbr_steps):
    
    rewards = []
    actions = []
    
    time_step = environment.reset()
    
    for _ in range(nbr_steps):

        action_step = policy.action(time_step)    
        actions.append(action_step.action.numpy()[0])
        time_step = environment.step(action_step.action)
        rewards.append(time_step.reward.numpy()[0])
    
    return actions, rewards


#%% -------Data collection-------
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

#%% -------
