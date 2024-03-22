import gym
import gym_donkeycar
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.regularizers import l2
from tensorflow.keras import initializers
import matplotlib.pyplot as plt


class OUActionNoise:
	def __init__(self, mean, std_deviation, theta, mult, dt=1e-1, decay = 1e-5, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.decay = decay
		self.mult = mult
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		x = self.mult*(
			self.x_prev
			+ self.theta * (self.mean - self.x_prev) * self.dt
			+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
		)
		self.std_dev = max(self.std_dev * (1 - self.decay), 0)
		print(self.std_dev)
		
		#Store x into x_prev
		#Makes next noise dependent on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)




#This update target parameters slowly
#Based on rate 'tau', which is much less than one
@tf.function
def update_target(target_weights, weights, tau):
	for (a, b) in zip(target_weights, weights):
		a.assign(b*tau+a*(1-tau))


def get_actor(num_states):
	# Initialize weights between -3e-3 and 3-e3
	th_init = tf.keras.initializers.GlorotUniform()

	inputs = layers.Input(shape=(num_states,),name="input")
	
	fc1 = layers.Dense(100,name="fc1",kernel_regularizer=l2(1e-4),activity_regularizer=l2(1e-4))(inputs)
	fc1 = layers.Activation('relu')(fc1)
	
	fc2 = layers.Dense(100,name="fc2",kernel_regularizer=l2(1e-4),activity_regularizer=l2(1e-4))(fc1)
	fc2 = layers.Activation('relu')(fc2)
	
	fc3 = layers.Dense(100,name="fc3",kernel_regularizer=l2(1e-4),activity_regularizer=l2(1e-4))(fc2)
	fc3 = layers.Activation('relu')(fc3)
		
	outputs_steer = layers.Dense(1,name="output_steer",kernel_initializer=th_init,activity_regularizer=l2(1e-4))(fc3)
	outputs_steer = layers.Activation('tanh')(outputs_steer)
	outputs_acc = layers.Dense(1,name="output_acc",kernel_initializer=th_init,activity_regularizer=l2(1e-4))(fc3)
	outputs_acc = layers.Activation('tanh')(outputs_acc)
	outputs_steer = outputs_steer * 0.2793
	outputs_acc = outputs_acc * 0.06
	outputs = layers.Concatenate()([outputs_steer, outputs_acc])
	model = tf.keras.Model(inputs, outputs)
	return model


def get_critic(num_states, num_actions):
	# State as input
	
	state_input = layers.Input(shape=(num_states),name="Observation")
	state_output = layers.Dense(100,activation="relu",name="fc1",kernel_regularizer=l2(1e-4))(state_input)
	state_output = layers.Dense(100,name="fc2",kernel_regularizer=l2(1e-4))(state_output)

	# Action as input
	action_input = layers.Input(shape=(num_actions),name="Actions")
	action_output = layers.Dense(100,name="fc3",kernel_regularizer=l2(1e-4))(action_input)

	# Both are passed through seperate layer before concatenating
	concat = layers.Add()([state_output, action_output])
	concat = layers.Activation('relu')(concat)
	conc_out = layers.Dense(100,activation="relu",name="fc4",kernel_regularizer=l2(1e-4))(concat)
	outputs = layers.Dense(1,activation="linear",name="QValue",kernel_regularizer=l2(1e-4))(conc_out)

	# Outputs single value for give state-action
	model = tf.keras.Model([state_input, action_input], outputs)

	return model

def policy(state, noise_object_st, noise_object_th, actor_model, lower_bound_steer, upper_bound_steer, lower_bound_acc, upper_bound_acc):
	sampled_actions = tf.squeeze(actor_model(state))
	print("Sampled Actions: {}".format(sampled_actions))
	noise_st = noise_object_st()
	noise_th = noise_object_th()
	#Adding noise to action
	# sampled_actions[0] = sampled_actions.numpy()[0] + noise
	sampled_actions = sampled_actions.numpy()
	sampled_actions[0] = sampled_actions[0] + noise_st
	sampled_actions[1] = sampled_actions[1] + noise_th
	legal_action = np.zeros(2)
	#We make sure action is within bounds
	legal_action[0] =  np.clip(sampled_actions[0], lower_bound_steer, upper_bound_steer)
	legal_action[1] = np.clip(sampled_actions[1], lower_bound_acc, upper_bound_acc)
	print("Bounded Actions: {}".format(legal_action))
	return [np.squeeze(legal_action)]
