import math as m
import numpy as np


def Rotation_Matrix(theta):
	R = np.array([[m.cos(theta), m.sin(theta)],[-m.sin(theta), m.cos(theta)]])
	return R

def alpha(dkr):
	a_r = 2*m.asin(0.5*dkr)
	return a_r

def error(rot_mat, long, lat, long_r, lat_r, d):
	z = np.dot(rot_mat, np.array([[long_r - long,lat_r - lat]]).T) - d
	return (z)

def steer_angle(steering):
	real_angle = np.radians(-16 + (16 * (steering + 1)))
	return real_angle
