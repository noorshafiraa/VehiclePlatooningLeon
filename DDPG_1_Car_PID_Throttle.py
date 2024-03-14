"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""
import argparse
import uuid
import timeit
import time
import gym
import gym_donkeycar
import numpy as np
import random
import matplotlib.pyplot as plt
from simple_pid import PID

if __name__ == "__main__":

	# Initialize the donkey environment
	# where env_name one of:
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

	parser = argparse.ArgumentParser(description="ppo_train")
	parser.add_argument(
		"--sim",
		type=str,
		default= "D:\Kuliah\S2\S2_Proposal_Thesis\DonkeyProject\DonkeySimWin\donkey_sim.exe",
		# default= "D:\Tesis\DonkeySimWin\donkey_sim.exe",
		help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
	)
	parser.add_argument("--port", type=int, default=9093, help="port to use for tcp")
	parser.add_argument("--test", action="store_true", help="load the trained model and play")
	parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
	parser.add_argument(
		"--env_name", type=str, default="donkey-generated-roads-v0", help="name of donkey sim environment", choices=env_list
	)

	args = parser.parse_args()

	if args.sim == "sim_path" and args.multi:
		print("you must supply the sim path with --sim when running multiple environments")
		exit(1)

	env_id = args.env_name

	conf = {
		"exe_path": args.sim,
		"host": "127.0.0.1",
		"port": args.port,
		"body_style": "donkey",
		"body_rgb": (128, 128, 128),
		"car_name": "leader",
		"font_size": 60,
		"racer_name": "PPO",
		"country": "ID",
		"bio": "Learning to drive w PPO RL",
		"guid": str(uuid.uuid4()),
		"max_cte": 10,
		"msg_type": "set_position",
		"pos_x" : "0.0",
		"pos_y" : "0.0",
		"pos_z" : "0.0",
		
	}
	if args.test:
		env = gym.make(args.env_name, conf=conf)
		for i in range(0,21):
			steer = 0
			with open('route{}.txt'.format(i), 'w+') as file:
					file.write("Steering,Throttle,Speed,PosZ,PosX,Yaw,Car_Angle")
			# Make an environment test our trained policy
			# model = PPO.load("ppo_donkey_gen3_3")

			obs = env.reset()
			pid = PID(10,0.5,0.1, setpoint=1)
			v_set = pid.setpoint
			pid.output_limits= (0.0,0.45)
			a = 0.01
			# ghost_data = np.genfromtxt('meta_data_test.txt',delimiter=',',encoding='utf-8-sig',skip_header=1)
			env.render()
			step = 0
			while step <= 600:
				print("Step: ",step)
				start = timeit.default_timer()
				# print(ghost_data[step,0:2])
				# steer = min(1,max(-1,steer+random.uniform(-0.2,0.2))) if step >= 10 else 0
				steer = 0.0
				obs, reward, done, info = env.step([steer,a])
				# print("Position X: ", info['pos'][0])
				# print("Position Z: ", info['pos'][2])
				v_out = info['speed']
				print("Speed: ", v_out)
				a = pid(v_out)
				print("Hit: ", info['hit'])
				print("Accel: ", a)
				meta_data = steer,a,info['speed'],info['pos'][2],info['pos'][0],info['gyro'][2],info['car'][2]
				with open('route{}.txt'.format(i), 'a') as file:
					file.write('\n')
					file.write(', '.join(map(str,meta_data)))
				step+=1
				# if random.random() < 0.3:
					# print("Turning")
					# opt = random.random()
					# for curve in range(20):
						# steer = 0.2 if opt < 0.5 else -0.2
						# prev = steer
						# obs, reward, done, info = env.step([steer,a])
						# v_out = info['speed']
						# a = pid(10*v_out)
						# meta_data = steer,a,info['speed'],info['pos'][2],info['pos'][0],info['gyro'][2],info['car'][2]
						# with open('route{}.txt'.format(i), 'a') as file:
							# file.write('\n')
							# file.write(str(meta_data))
							# file.write(', '.join(map(str,meta_data)))
					# step+=20
					# for curve in range(20):
						# steer = -0.2 if prev == 0.2 else 0.2
						# obs, reward, done, info = env.step([steer,a])
						# v_out = info['speed']
						# a = pid(10*v_out)
						# meta_data = steer,a,info['speed'],info['pos'][2],info['pos'][0],info['gyro'][2],info['car'][2]
						# with open('route{}.txt'.format(i), 'a') as file:
							# file.write('\n')
							# file.write(str(meta_data))
							# file.write(', '.join(map(str,meta_data)))
					# step+=20
				# else:
					# step += 1
					# meta_data = steer,a,info['speed'],info['pos'][2],info['pos'][0],info['gyro'][2],info['car'][2]
					# with open('route{}.txt'.format(i), 'a') as file:
						# file.write('\n')
						# file.write(str(meta_data))
						# file.write(', '.join(map(str,meta_data)))

	# if args.test:
		# plot_speed = []
		# plt_speed = np.zeros([10,20])
		# # Make an environment test our trained policy
		# env = gym.make(args.env_name, conf=conf)
		# model = PPO.load("ppo_donkey_gen3_2")
		# throttle = 0.1
		# obs = env.reset()
		# for ep in range(10):
			# action = [0 , throttle + (0.1 * ep)]
			# for _ in range(20):
				# start = timeit.default_timer()
				# # action, _states = model.predict(obs, deterministic=True)
				# obs, reward, done, info = env.step(action)
				# print("Speed: ", info['speed'])
				# plot_speed.append(info['speed'])
				# env.render()
				# stop = timeit.default_timer()
				# print("Time: ", stop-start)
			# # plot_speed = np.array(plot_speed)
			# # np.savetxt('plot_speed{:03}.txt'.format(ep),delimiter=',')
			# obs = env.reset()
			# print("Speed: ", plot_speed)
			# print("Torque used: ", action[1])
			# plt.plot(plot_speed)
			# plt_speed[ep] = plot_speed
			# plot_speed = []
			# print(plt_speed)
		# np.savetxt('plot_speed.txt',plt_speed,delimiter=',')
		# plt.show()
		# print("done testing")
		
	else:

		# make gym env
		env = gym.make(args.env_name, conf=conf)

		# create cnn policy
		model = PPO("CnnPolicy", env, verbose=1)

		# set up model in learning mode with goal number of timesteps to complete
		model.learn(total_timesteps=100000)

		obs = env.reset()

		for i in range(1000):

			action, _states = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)
			try:
				env.render()
			except Exception as e:
				print(e)
				print("failure in render, continuing...")

			if done:
				obs = env.reset()

			if i % 100 == 0:
				print("saving...")
				model.save("ppo_donkey_gen3_3")

		# Save the agent
		model.save("ppo_donkey_gen3_3")
		print("done training")
	time.sleep(1)
	env.close()
