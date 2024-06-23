"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
Modified for CACC by : Leon Wonohito
Date = 29 July 2022
"""
import argparse
import uuid
import os

import gym
import gym_donkeycar
import Class_Train_DDPG_path_optimize
import State_Cal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math as m
import random

class Buffer:
	def __init__(self, buffer_capacity=int(1e6), batch_size=64, num_states=9, num_actions=2): #Previously 64 batch size 50000 buffer cap
		#Number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		#Num of tuples to train on
		self.batch_size = batch_size

		#Number of times record() was called
		self.buffer_counter = 0

		#Instead of list of tuples as the exp.replay concept go
		#We use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

	#Takes (s,a,r,s') observation tuple as input
	def record(self, obs_tuple):
		#Set index to zero if buffer_capacity is exceeded,
		#replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1

	# Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
	# TensorFlow to build a static graph out of the logic and computations in our function.
	# This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
	@tf.function
	def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
		#Training and updating Actor & Critic networks
		#See pseudocode
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_state_batch, training=True)
			y = reward_batch + gamma * target_critic(
				[next_state_batch, target_actions], training=True
			)
			critic_value = critic_model([state_batch, action_batch], training=True)
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
			

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch, training=True)
			critic_value = critic_model([state_batch, actions], training=True)
			#Used '-value' as we want to maximize the value given
			#by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

	#We compute the loss and update parameters
	def learn(self):
		#Get sampling range
		record_range = min(self.buffer_counter, self.buffer_capacity)
		
		#Randomly sampe indices
		batch_indices = np.random.choice(record_range, self.batch_size)

		#Convert to tensors
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
		
		self.update(state_batch, action_batch, reward_batch, next_state_batch)

if __name__ == "__main__":

	# Initialize the donkey environment
	env_list = [
		"donkey-warehouse-v0",
		"donkey-generated-roads-v0",
		"donkey-avc-sparkfun-v0",
		"donkey-generated-track-v0",
		"donkey-roboracingleague-track-v0",
		"donkey-waveshare-v0",
		"donkey-minimonaco-track-v0",
		"donkey-warren-track-v0",
		"donkey-thunderhill-track-v0",
		"donkey-circuit-launch-track-v0",
	]

	parser = argparse.ArgumentParser(description="thesis_train")
	parser.add_argument(
		"--sim",
		type=str,

        # There are two ways to open simulator :
        # use "remote" to open manually or 
        # declare your simulator path to open automatically
        default= "remote",
        # default= "<YOUR PATH>"

		help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
	)
	parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
	parser.add_argument("--train", action="store_true", help="load the trained model and play")
	parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
	parser.add_argument(
		"--env_name", type=str, default="donkey-generated-roads-v0", help="name of donkey sim environment", choices=env_list
	)

	args = parser.parse_args()

	if args.sim == "sim_path" and args.multi:
		print("you must supply the sim path with --sim when running multiple environments")
		exit(1)

	env_id = args.env_name
	
	conf2 = {
		"exe_path": args.sim,
		"host": "127.0.0.1",
		"port": args.port,
		"body_style": "donkey",
		"body_rgb": (128, 128, 128),
		"car_name": "follower",
		"font_size": 40,
		"racer_name": "DDPG",
		"country": "ID",
		"bio": "Follow the Leader",
		"guid": str(uuid.uuid4()),
		"max_cte": 10,
	}
	
	env2 = gym.make(args.env_name, conf=conf2)
	num_actions = env2.action_space.shape[0]

	upper_bound_steer = 0.2793
	lower_bound_steer = -0.2793
	upper_bound_acc = 0.45 * 0.06
	lower_bound_acc = 0.0 * 0.06
	std_dev_th = 0.6
	std_dev_st = 0.1
	
	ou_noise_th = Class_Train_DDPG_path_optimize.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev_th)* np.ones(1), theta=0.15, mult = 0.1)
	ou_noise_st = Class_Train_DDPG_path_optimize.OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev_st)* np.ones(1), theta=0.15, mult = 0.5)
	state_size = 9
	num_states = np.zeros(state_size).shape[0]
	
	if args.train:
		#Early Termination flag
		end = False
		
		if os.path.isfile("cacc_actor_loc_best.h5"):
			actor_model = keras.models.load_model("cacc_actor_loc_best.h5")
			critic_model = keras.models.load_model('cacc_critic_loc_best.h5')
			
			target_actor = keras.models.load_model('cacc_target_actor_loc_best.h5')
			target_critic = keras.models.load_model('cacc_target_critic_loc_best.h5')
			
			critic_lr = 1e-3
			actor_lr = 1e-4

			critic_optimizer = tf.keras.optimizers.Adam(critic_lr,clipnorm=1.0)
			actor_optimizer = tf.keras.optimizers.Adam(actor_lr,clipnorm=1.0)
			
			actor_model.compile(optimizer=actor_optimizer)
			critic_model.compile(optimizer=critic_optimizer)
			target_actor.compile(optimizer=actor_optimizer)
			target_critic.compile(optimizer=critic_optimizer)

		else:
			actor_model = Class_Train_DDPG_path_optimize.get_actor(num_states)
			critic_model = Class_Train_DDPG_path_optimize.get_critic(num_states, num_actions)

			target_actor = Class_Train_DDPG_path_optimize.get_actor(num_states)
			target_critic = Class_Train_DDPG_path_optimize.get_critic(num_states, num_actions)

			target_actor.set_weights(actor_model.get_weights())
			target_critic.set_weights(critic_model.get_weights())

			keras.utils.plot_model(actor_model, "Actor_Model_new.png", show_shapes=True, show_layer_activations=True,show_layer_names = True)
			keras.utils.plot_model(critic_model, "Critic_Model_new.png", show_shapes=True, show_layer_activations=True, show_layer_names = True)
			keras.utils.plot_model(target_actor, "Target_Actor_new.png", show_shapes=True, show_layer_activations=True)
			keras.utils.plot_model(target_critic, "Target_Critic_new.png", show_shapes=True, show_layer_activations=True)
			
			critic_lr = 1e-3
			actor_lr = 1e-4

			critic_optimizer = tf.keras.optimizers.Adam(critic_lr,clipnorm=1.0)
			actor_optimizer = tf.keras.optimizers.Adam(actor_lr,clipnorm=1.0)

		# Discount factor for future rewards
		gamma = 0.99
		# Used to update target networks
		tau = 1e-3
		buffer = Buffer(int(1e6), 64, num_states, num_actions)
		ep_reward_list = []
		avg_reward_list = []
		long_err_list = []
		lat_err_list = []
		long_err_max = []
		long_err_min = []
		lat_err_max = []
		lat_err_min = []
		step_max = []
		acc_in = []
		steer_in = []
		
		fol_pos_list = []
	
		episodic_reward = 0
		
		fig,axs = plt.subplots(nrows=4,ncols=1)
		axs[0].set(ylabel="Avg. Reward")
		axs[1].set(ylabel="Avg. Long.")
		axs[2].set(ylabel="Avg. Lat.")
		axs[3].set(xlabel="Episode", ylabel="Total Step")
		axs[0].grid()
		axs[1].grid()
		axs[2].grid()
		axs[3].grid()
		fig.tight_layout(h_pad=2)
		#Getting Initial Position of Follower
		##########################################################
		total_episode = 10000
		
		# init_lead_pos_data = ghost_data[:,4]
		from simple_pid import PID
		pid = PID(10,0.5,0.1, setpoint=1.0)
		v_set = pid.setpoint
		pid.output_limits = (-1,1)
		#Longitudinal is Z Lateral is X
		try:
			for ep in range(total_episode):
				path = np.random.randint(0,22)
				print("Path: {}".format(path))
				ghost_data = np.genfromtxt('route{}.txt'.format(path),delimiter=',',encoding='utf-8-sig',skip_header=1)
				init_lead_pos_data = ghost_data[:,3]
				step = 11
				state = env2.reset()
				prev_acc = 0.5
				prev_steer = 0
				v_out = 0
				while v_out < 1.0:
					prev_act = [prev_steer, prev_acc]
					_, _, _, info2 = env2.step(prev_act)
					v_out = info2['speed']
					prev_acc = pid(v_out)
				prev_act = [-0.3, prev_acc]
				_, _, _, info2 = env2.step(prev_act)
				init_rand = random.random()
				rand = random.random()
				
				prev_pos_follow = [info2['pos'][2] - 1, info2['pos'][0] + (init_rand) if rand > 0.5 else info2['pos'][0] - (init_rand)]

				if rand > 0.5:
					ghost_data[:,3] = init_lead_pos_data + (0.1*init_rand)
					
				else:
					ghost_data[:,3] = init_lead_pos_data - (0.1*init_rand)
				
				
				err_step_long = []
				err_step_lat = []
				i_v_err = []
				i_z = []
				episodic_reward = 0
				long_err_mean = 0
				lat_err_mean = 0
				#desired distance
				time_gap = 0.1
				r = 0.3
				v = info2['speed']
				dist = r + (time_gap*v)
				print("Distance: ", dist)
				
				init_speed_fol = info2['speed']

				kr = ghost_data[step-1,5]/ghost_data[step-1,2] if ghost_data[step-1,2] != 0 else 0
				sat_dkr = min(1,max(-1,dist*kr))
				alpr = State_Cal.alpha(sat_dkr)
				sa = 0.5*sat_dkr
				ca = 0.5*m.sqrt(4-sat_dkr**2)
				print(type(info2['car'][2]))
				fol_angle = info2['car'][2]
				if ghost_data[step-1,6] > 180:
					ghost_data[step-1,6] -= 360.0
				if info2['car'][2] > 180:
					fol_angle = fol_angle - 360.0
					
				epsi = np.radians(ghost_data[step-1,6]) - np.radians(fol_angle) - alpr
				d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
				z_prev	= (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(fol_angle)),prev_pos_follow[0], prev_pos_follow[1],
							ghost_data[step-1,3], ghost_data[step-1,4], d))
				
				kr = ghost_data[step,5]/ghost_data[step,2] if ghost_data[step,2] != 0 else 0
				sat_dkr = min(1,max(-1,dist*kr))
				alpr = State_Cal.alpha(sat_dkr)
				sa = 0.5*sat_dkr
				ca = 0.5*m.sqrt(4-sat_dkr**2)
				
				if ghost_data[step,6] > 180:
					ghost_data[step,6] -= 360
				
				epsi = np.radians(ghost_data[step,6]) - np.radians(fol_angle) - alpr
				d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
				lat = info2['pos'][0] + (init_rand) if rand > 0.5 else info2['pos'][0] - (init_rand)
				print(lat)
				print(info2['pos'][2] - 1)
				z	= (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(fol_angle)), (info2['pos'][2] - 1), lat,
							ghost_data[step,3], ghost_data[step,4], d))
				i_z.append(z)
				z_dot = (z - z_prev)/0.017
				integ_z = 0.017 * (sum(i_z))
				v_err = ghost_data[step,2] - info2['speed']
				v_err_prev = ghost_data[step-1,2] - 0
				integ_v_err = 0.017*(v_err_prev + v_err)
				i_v_err.append(v_err)
				state = np.array(np.concatenate((np.array([integ_v_err]), np.array([v_err]), np.array([info2['speed']]), integ_z[0], z[0], integ_z[1], z[1], z_dot.flatten())))
				z_prev = z
				v_err_prev = v_err
				print("Initial State: ", state)
				step+=1

				try:
					while True:
						# env2.render()
						print("Step: ", step)
						
						#Calculate some values from previous input
						steer_input = (prev_act[0]/0.2793)**2
						acc_input = (prev_act[1]/0.06)**2
						#Leader Control in Ghost Data
						pos_lead = ghost_data[step,3:5] #PosZ,PosX
						lead_angle = State_Cal.steer_angle(ghost_data[step,0])
						
						#Follower Control
						tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
						action2 = Class_Train_DDPG_path_optimize.policy(tf_prev_state, ou_noise_st, ou_noise_th, actor_model, lower_bound_steer, upper_bound_steer,
									lower_bound_acc, upper_bound_acc)
						action2[0][0] = action2[0][0]/0.2793
						action2[0][1] = action2[0][1]/0.06
						if action2[0][1] < 0 and info2['vel'][2] < 0:
							info2['speed'] = -info2['speed']
						print("Steering: {} -- Throttle: {}".format(np.degrees(State_Cal.steer_angle(action2[0][0])), action2[0][1]))
						_, _, _, info2 = env2.step(action2[0])
						prev_act = action2[0]
						
						#Find the next state
						pos_follow = [info2['pos'][2] - 1, info2['pos'][0] + (init_rand) if rand > 0.5 else info2['pos'][0] - (init_rand)] #PosZ,PosX
						print("Follower's Position -> ",pos_follow)
						print("Leader's Position ->", pos_lead)
						v = info2['speed']
						print("Speed {}".format(v))
						print("Velocity Longitude: {}".format(info2['vel'][2]))
						dist = r + (time_gap*v)
						print("Distance: ", dist)
						kr = ghost_data[step,5]/ghost_data[step,2] if ghost_data[step,2] != 0 else 0
						sat_dkr = min(1,max(-1,dist*kr))
						alpr = State_Cal.alpha(sat_dkr)
						sa = 0.5*sat_dkr
						ca = 0.5*m.sqrt(4-sat_dkr**2)
						fol_angle = info2['car'][2]
						if ghost_data[step,6] > 180:
							ghost_data[step,6] -= 360
						if fol_angle > 180:
							fol_angle = fol_angle - 360.0
							
						epsi = np.radians(ghost_data[step,6]) - np.radians(fol_angle) - alpr
						d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
						
						z = (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(fol_angle)),pos_follow[0], pos_follow[1],
							pos_lead[0], pos_lead[1], d))
						i_z.append(z)
						err_step_long.append(z[0])
						err_step_lat.append(z[1])
						z_dot = (z - z_prev)/0.017
						integ_z = 0.017 * (sum(i_z))
						v_err = ghost_data[step,2] - v
						i_v_err.append(v_err)
						integ_v_err = 0.017*(sum(i_v_err))

						new_state = np.array(np.concatenate((np.array([integ_v_err]), np.array([v_err]), np.array([info2['speed']]), integ_z[0], z[0], integ_z[1], z[1], z_dot.flatten())))
						z_prev = z
						v_err_prev = v_err
						print("New State: ",new_state)
						#Calculate reward
						vel_err_in = ghost_data[step,2] - info2['speed']
						lat_err = z[1]**2
						long_err = z[0]**2
						vel_err = (vel_err_in)**2
						# F = 1 if done2 else 0
						H = 1 if (z[1] < 0.1) and (z[1] > -0.1) else 0
						M = 1 if (z[0] < 0.1) and (z[0] > -0.1) else 0
						F = 0
						reward2 = (-((50 * long_err) + (5 * steer_input) + (50 * lat_err) + (5 * acc_input) + (int(ghost_data.shape[0])-1 - step)) * 1e-3
						- 10*F + 10*H + 10*M )
						
						#Learn Block
						buffer.record((state, prev_act, reward2, new_state))
						episodic_reward += reward2
						long_err_mean += long_err
						lat_err_mean += lat_err
						buffer.learn()
						
						Class_Train_DDPG_path_optimize.update_target(target_actor.variables, actor_model.variables, tau)
						Class_Train_DDPG_path_optimize.update_target(target_critic.variables, critic_model.variables, tau)

						#New State Condition
						state = new_state
						step += 1
						
						if step == (int(ghost_data.shape[0])-1) or m.sqrt(long_err) > 5 or m.sqrt(lat_err) > 5 or info2['vel'][2] < -0.5 or ((abs(pos_lead[0] - pos_follow[0]) <= 0.2) and (abs(pos_lead[1] - pos_follow[1] <= 0.2))):
							F = 1
							#Crash Constant CC
							CC = 100
							reward2 = (-((50 * long_err) + (5 * steer_input) + (50 * lat_err) + (5 * acc_input) + (int(ghost_data.shape[0])-1 - step)) * 1e-3
										- 10*F + 10*H + 10*M )
							buffer.record((state, prev_act, reward2, new_state))
							episodic_reward += reward2
							long_err_mean += long_err
							lat_err_mean += lat_err
							buffer.learn()
							
							Class_Train_DDPG_path_optimize.update_target(target_actor.variables, actor_model.variables, tau)
							Class_Train_DDPG_path_optimize.update_target(target_critic.variables, critic_model.variables, tau)
							break
				except KeyboardInterrupt:
					actor_model.save("cacc_actor_new.h5")
					critic_model.save("cacc_critic_new.h5")

					target_actor.save("cacc_target_actor.h5")
					target_critic.save("cacc_target_critic.h5")
					end = True
					break
					
				ep_reward_list.append(episodic_reward)
				long_err_list.append(long_err_mean/step)
				long_err_max.append(np.max(err_step_long))
				long_err_min.append(np.min(err_step_long))
				lat_err_list.append(lat_err_mean/step)
				lat_err_max.append(np.max(err_step_lat))
				lat_err_min.append(np.min(err_step_lat))
				step_max.append(step)
				
				# Mean of last 40 episodes
				avg_reward = np.mean(ep_reward_list[-40:])
				print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
				avg_reward_list.append(avg_reward)

				if avg_reward >= 8000 and step >= 450:
					if (lat_err_mean/step <= 0.3) and (long_err_mean/step <= 0.3):
						actor_model.save("cacc_actor_best.h5")
						critic_model.save("cacc_critic_best.h5")

						target_actor.save("cacc_target_actor_best.h5")
						target_critic.save("cacc_target_critic_best.h5")
						break
					else:
						actor_model.save("cacc_actor_loc_best.h5")
						critic_model.save("cacc_critic_loc_best.h5")

						target_actor.save("cacc_target_actor_loc_best.h5")
						target_critic.save("cacc_target_critic_loc_best.h5")
				if ep%1000 == 0:
					plt.ion()
					print("Temp Results")
					axs[0].plot(avg_reward_list,'b')
					axs[1].plot(long_err_list,'b',label="Avg.")
					axs[1].plot(long_err_max,'r',alpha=0.5,label="Max R")
					axs[1].plot(long_err_min,'g',alpha=0.5,label="Max L")
					
					axs[2].plot(lat_err_list,'b',label="Avg.")
					axs[2].plot(lat_err_max,'r',alpha=0.5,label="Max R")
					axs[2].plot(lat_err_min,'g',alpha=0.5,label="Max L")
					
					axs[3].plot(step_max,'b')
					if ep == 0:
						axs[1].legend(loc="lower left")
						axs[2].legend(loc="lower left")
					
					plt.draw()
					plt.pause(.001)
		except KeyboardInterrupt:
			env2.close()
			if os.path.isfile("cacc_actor_new.h5"):
				pass
			else:
				actor_model.save("cacc_actor_new.h5")
				critic_model.save("cacc_critic_new.h5")

				target_actor.save("cacc_target_actor.h5")
				target_critic.save("cacc_target_critic.h5")
		env2.close()

		# fig,axs = plt.subplots(nrows=4,ncols=1)
		
		axs[0].plot(avg_reward_list)
		axs[1].plot(long_err_list,label="Avg.")
		axs[1].plot(long_err_max,alpha=0.5,label="Max R")
		axs[1].plot(long_err_min,alpha=0.5,label="Max L")
		axs[2].plot(lat_err_list,label="Avg.")
		axs[2].plot(lat_err_max,alpha=0.5,label="Max R")
		axs[2].plot(lat_err_min,alpha=0.5,label="Max L")
		axs[3].plot(step_max)

		fig.savefig("DDPG_Train Model.eps")
		fig.savefig("DDPG_Train_Model.png")
		plt.show()

		actor_model.save("cacc_actor_new.h5")
		critic_model.save("cacc_critic_new.h5")

		target_actor.save("cacc_target_actor.h5")
		target_critic.save("cacc_target_critic.h5")
	
	else:
		if os.path.isfile("cacc_actor_best.h5"):
			print("Load the best result") #Steer 0.07719711 Acc 0.46318268
			model = keras.models.load_model("cacc_actor_best.h5")
		elif os.path.isfile("cacc_actor_loc_best.h5"):
			print("Load good enough result!!")
			model = keras.models.load_model("cacc_actor_loc_best.h5")
		else:
			print("Load ANY result")
			model = keras.models.load_model("cacc_actor_new.h5")

		actor_lr = 1e-4
		actor_optimizer = tf.keras.optimizers.Adam(actor_lr,clipnorm=1.0)
		model.compile(optimizer=actor_optimizer)
		
		ghost_data = np.genfromtxt('route_test_1.txt',delimiter=',',encoding='utf-8-sig',skip_header=1)
		
		ou_noise_th = Class_Train_DDPG_path_optimize.OUActionNoise(mean=np.zeros(1), std_deviation=float(0)* np.ones(1), theta=0.15, mult = 0.1)
		ou_noise_st = Class_Train_DDPG_path_optimize.OUActionNoise(mean=np.zeros(1), std_deviation=float(0)* np.ones(1), theta=0.15, mult = 0.5)
		state = env2.reset()
		
		from simple_pid import PID
		pid = PID(10,0.5,0.1, setpoint=1.0)
		v_set = pid.setpoint
		pid.output_limits = (-1,1)
		prev_acc = 0.5
		prev_steer = 0
		v_out = 0
		while v_out < 1.0:
			prev_act = [prev_steer, prev_acc]
			_, _, _, info2 = env2.step(prev_act)
			v_out = info2['speed']
			prev_acc = pid(v_out)
		prev_act = [0.0, prev_acc]
		_, _, _, info2 = env2.step(prev_act)
		prev_pos_follow = [info2['pos'][2] - 1, info2['pos'][0]]
		z_long = []
		z_lat = []
		i_v_err = []
		i_z = []
		pos_map = []
		episodic_reward = 0
		long_err_mean = 0
		lat_err_mean = 0
		step = 11
		
		#desired distance
		time_gap = 0.1
		r = 0.3
		v = info2['speed']
		dist = r + (time_gap*v)
		print("Distance: ", dist)
		with open("pos_data_new.txt","a") as file:
			file.write(str(prev_pos_follow))
			file.write('\n')
		init_speed_fol = info2['speed']

		kr = ghost_data[step-1,5]/ghost_data[step-1,2] if ghost_data[step-1,2] != 0 else 0
		sat_dkr = min(1,max(-1,dist*kr))
		alpr = State_Cal.alpha(sat_dkr)
		sa = 0.5*sat_dkr
		ca = 0.5*m.sqrt(4-sat_dkr**2)
		
		epsi = np.radians(ghost_data[step-1,-1]) - np.radians(info2['car'][2]) + alpr
		d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
		z_prev	= (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(info2['car'][2])),prev_pos_follow[0], prev_pos_follow[1],
					ghost_data[step-1,3], ghost_data[step-1,4], d))
		
		kr = ghost_data[step,5]/ghost_data[step,2] if ghost_data[step,2] != 0 else 0
		sat_dkr = min(1,max(-1,dist*kr))
		alpr = State_Cal.alpha(sat_dkr)
		sa = 0.5*sat_dkr
		ca = 0.5*m.sqrt(4-sat_dkr**2)
		
		epsi = np.radians(ghost_data[step,-1]) - np.radians(info2['car'][2]) + alpr
		d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
		
		z	= (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(info2['car'][2])), info2['pos'][2] - 1, info2['pos'][0],
					ghost_data[step,3], ghost_data[step,4], d))
		i_z.append(z)
		z_dot = (z - z_prev)/0.017
		integ_z = 0.017 * (sum(i_z))
		v_err = ghost_data[step,2] - info2['speed']
		v_err_prev = ghost_data[step-1,2] - 0
		integ_v_err = 0.017*(v_err_prev + v_err)
		i_v_err.append(v_err)
		state = np.array(np.concatenate((np.array([integ_v_err]), np.array([v_err]), np.array([info2['speed']]), integ_z[0], z[0], integ_z[1], z[1], z_dot.flatten())))
		z_prev = z
		v_err_prev = v_err
		print("Initial State: ", state)
		step+=1
		for i in range(step, len(ghost_data)-1):
			print("Step {}".format(i))
			pos_lead = ghost_data[i,3:5]
			tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
			action2 = Class_Train_DDPG_path_optimize.policy(tf_prev_state, ou_noise_st, ou_noise_th, model, lower_bound_steer, upper_bound_steer,
									lower_bound_acc, upper_bound_acc)
			action2[0][0] = action2[0][0]/0.2793
			action2[0][1] = action2[0][1]/0.06
			if action2[0][1] < 0 and info2['vel'][2] < 0:
				info2['speed'] = -info2['speed']
			print("Steering: {} -- Throttle: {}".format(np.degrees(State_Cal.steer_angle(action2[0][0])), action2[0][1]))
			_, _, _, info2 = env2.step(action2[0])
			
			pos_follow = [info2['pos'][2] - 1, info2['pos'][0]] #PosZ,PosX
			print("Follower's Position -> ",pos_follow)
			print("Leader's Position ->", np.array(pos_lead))
			v = info2['speed']
			dist = r + (time_gap*v)
			print("Distance: ", dist)
			kr = ghost_data[i,5]/ghost_data[i,2] if ghost_data[i,2] != 0 else 0
			sat_dkr = min(1,max(-1,dist*kr))
			alpr = State_Cal.alpha(sat_dkr)
			sa = 0.5*sat_dkr
			ca = 0.5*m.sqrt(4-sat_dkr**2)
			
			epsi = np.radians(ghost_data[i,6]) - np.radians(info2['car'][2]) + alpr
			d = dist*np.dot(State_Cal.Rotation_Matrix(epsi).T,np.array([[ca,sa]]).T)
			
			z = (State_Cal.error(State_Cal.Rotation_Matrix(np.radians(info2['car'][2])),pos_follow[0], pos_follow[1],
				pos_lead[0], pos_lead[1], d))
			i_z.append(z)
			z_dot = (z - z_prev)/0.017
			integ_z = 0.017 * (sum(i_z))
			v_err = ghost_data[step,2] - v
			i_v_err.append(v_err)
			integ_v_err = 0.017*(sum(i_v_err))

			state = np.array(np.concatenate((np.array([integ_v_err]), np.array([v_err]), np.array([info2['speed']]), integ_z[0], z[0], integ_z[1], z[1], z_dot.flatten())))
			z_prev = z
			v_err_prev = v_err
			z_long.append(z[0])
			z_lat.append(z[1])
			plt.plot(z_long,'b')
			plt.plot(z_lat,'r')
			with open("speed_data_new.txt","a") as file:
				file.write(str(v_err))
				file.write('\n')
			with open("err_data_new.txt","a") as file:
				file.write(str(z[0])+','+str(z[1]))
				file.write('\n')
			with open("pos_data_new.txt","a") as file:
				file.write(str(pos_follow))
				file.write('\n')
	plt.legend(loc="lower left")
	plt.title("Test Result")
	plt.show()
	env2.close()
