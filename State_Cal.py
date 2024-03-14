import math as m
import numpy as np


def Rotation_Matrix(theta):
	R = np.array([[m.cos(theta), m.sin(theta)],[-m.sin(theta), m.cos(theta)]])
	return R

def alpha(dkr):
	a_r = 2*m.asin(0.5*dkr)
	return a_r

def xys(pos_lead, d, a_r, rot_mat):
	arr = np.array([1-m.cos(a_r/2),-m.sin(a_r/2)])
	return (pos_lead.T + (d*np.dot(rot_mat,arr.T)))

def error(rot_mat, long, lat, long_r, lat_r, d):
	z = np.dot(rot_mat, np.array([[long_r - long,lat_r - lat]]).T) - d
	return (z)

def steer_angle(steering):
	real_angle = np.radians(-16 + (16 * (steering + 1)))
	return real_angle
	
def error_dot(d, alpha_r, yaw_fol, yaw_lead, err, speed_fol, speed_lead, lead_steering, fol_steering):
	#Asumsi dahulu alpha_dot = 0 karena curve_dot(kappa_dot) = 0 (Pers. 4.7)
	alpha_dot = 0
	yaw_lead = np.radians(yaw_lead)
	yaw_fol = np.radians(yaw_fol)
	delta = yaw_fol - yaw_lead + alpha_r
	e_dot = ((lead_steering - alpha_dot) * np.dot(np.array([[0,1],[1,0]]),err) - np.array([[speed_lead,d*lead_steering]]).T
			+ np.dot(np.array([[m.cos(delta),-d*m.sin(delta)],[m.sin(delta),d*m.cos(delta)]]),np.array([[speed_fol,fol_steering]]).T))
	return (e_dot, delta)
	
def local_pos(long_prev, v_now):
	t = 0.1
	return (long_prev + (v_now*t))