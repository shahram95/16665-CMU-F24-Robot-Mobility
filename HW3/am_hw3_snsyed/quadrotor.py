"""
Air Mobility Project- 16665
Author: Guanya Shi
guanyas@andrew.cmu.edu
"""
import sys
import numpy as np
import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as tf
import matplotlib.pyplot as plt
from time import sleep
from math_utils import *
from scipy.spatial.transform import Rotation
import argparse
np.random.seed(0)
def compute_rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual)**2))

class Quadrotor():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = np.diag([8.571710e-5, 8.655602e-5, 15.261652e-5]) # inertia matrix
		self.J_inv = np.linalg.inv(self.J)
		self.arm = 0.0325 # arm length
		self.t2t = 0.006 # thrust to torque ratio
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1., 1., 1., 1.],
			               [-self.arm, -self.arm, self.arm, self.arm],
			               [-self.arm, self.arm, self.arm, -self.arm],
			               [-self.t2t, self.t2t, -self.t2t, self.t2t]])
		self.B_inv = np.linalg.inv(self.B)
		
		# noise level
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		
		# disturbance and its estimation
		self.d = np.array([0., 0, 0])
		self.d_hat = np.array([0., 0, 0])

		# initial state
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])

		# initial control (hovering)
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.012 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		self.d = np.array([0., 0, 0])
		self.p = np.array([0., 0, 0])
		self.v = np.array([0., 0, 0])
		self.R = np.eye(3)
		self.q = np.array([1., 0, 0, 0])
		self.omega = np.array([0., 0, 0])
		self.euler_rpy = np.array([0., 0, 0])
		self.u = np.array([1, 1, 1, 1]) * self.m * self.g / 4.
		self.step = 0
		self.t = 0.

	def dynamics(self, u):
		'''
		Problem B-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (four rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		Hint: use the qintegrate function to update self.q
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)

		# Convert rotor forces to total thrust and torque
		thrust_torque = self.B @ self.u
		
		# Translational Dynamics
		total_thrust = thrust_torque[0]  # Total thrust is the first element
		
		pdot = self.v

		gravity_force = np.array([0, 0, -self.m * self.g])
		net_force = gravity_force + self.R @ np.array([0, 0, total_thrust])
		vdot = net_force / self.m

		# Rotational Dynamics using Euler's equations
		torque = thrust_torque[1:]  # The remaining three are the torques
		omegadot = self.J_inv @ (torque - np.cross(self.omega, self.J @ self.omega))
		
		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * (self.sigma_t * np.random.normal(size=3) + self.d) 
		self.q = self.q = qintegrate(self.q, self.omega, self.dt)  # This assumes that qintegrate updates quaternion based on angular velocity
		self.R = qtoR(self.q)
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal(size=3)
		self.euler_rpy = Rotation.from_matrix(self.R).as_euler('xyz')

		self.t += self.dt
		self.step += 1

	def S_inv(matrix):
		return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

	def cascaded_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-2: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a cascaded controller to track a trajectory (p_d, v_d, a_d, yaw_d).
		Hint for gain tuning: position control gain is smaller (1-10);
		Attitude control gain is bigger (10-200).
		'''
		# position control
		K_P = 2.5
		K_D = 2.5

		e_p = self.p - p_d
		e_v = self.v - v_d
		
		f_d = self.g*np.array([0,0,1]) - K_P * e_p - K_D * e_v + a_d
		z = self.R @ np.array([0, 0, 1])
		T = np.dot(f_d.T, z) * self.m
		
		z_d = f_d / np.linalg.norm(f_d)
		e3 = np.array([0, 0, 1])
		n = np.cross(e3, z_d)     # Compute rotation axis
		rho = np.arcsin(np.linalg.norm(n))  # Compute rotation angle
		
		n_normalized = n / np.linalg.norm(n)
		rotation_vector = rho * n_normalized
		REB = Rotation.from_rotvec(rotation_vector).as_matrix()  # Get the matrix representation of the rotation

		RAE = Rotation.from_euler('z', yaw_d).as_matrix()

		Rd = RAE @ REB

		# attitude control
		K_Ptau = 200
		K_Dtau = 20
		
		# Compute Re using given formula
		Re = Rd.T @ self.R
		
		R_diff = Re-Re.T
		S_inv = np.array([-R_diff[1,2], R_diff[0,2], -R_diff[0,1]])
		
		# Desired angular velocities are 0 (as given in slide)
		omega_d = np.array([0, 0, 0])
		alpha_d = np.array([0, 0, 0])  # Assumed to be zero unless specified

		
		# Control input for attitude control using the formula from the slide
		alpha = -K_Ptau * S_inv - K_Dtau * (self.omega - omega_d) + alpha_d
		
		J_alpha = np.dot(self.J, alpha)
		omega_cross_omega = np.cross(np.dot(self.J,self.omega), self.omega)
		tau = J_alpha - omega_cross_omega
		

		u = self.B_inv @ np.concatenate((np.array([T]),tau))
		
		return u.flatten()
	
	# def adaptive_control(self, p_d, v_d, a_d, yaw_d):
	# 	'''
	# 	Problem B-3: Based on lecture 4, implement adaptive control methods.
	# 	For integral control, this function should be same as cascaded_control, 
	# 	with an extra I gain in the position control loop.
	# 	Hint for integral control: you can use self.d_hat to accumlate/integrate the position error.
	# 	'''
	# 	'''if not hasattr(self, 'd_hat'):
	# 		self.d_hat = np.zeros_like(self.p)
	# 	if not hasattr(self, 'v_hat'):
	# 		self.v_hat = np.zeros_like(self.v)
	# 	'''
		
	# 	# position control
	# 	K_P = 2.5
	# 	K_D = 2.5

	# 	e_p = self.p - p_d
	# 	e_v = self.v - v_d
	# 	#self.d_hat += np.array(e_p)
	# 	self.d_hat = robot.d
	# 	f_d = self.g*np.array([0,0,1]) - K_P * e_p - K_D * e_v + a_d - self.d_hat
		
	# 	#z = self.R @ np.array([0, 0, 1])
	# 	#T = np.dot(f_d.T, z) * self.m

	# 	# Parameters (you can tweak these values)
	# 	#alpha = 0  # For the low-pass filter
	# 	As = -0.5  # Assuming it's a negative scalar as suggested by the slide
	# 	h = 0.01  # Discretization step, should be set according to your system
		
	# 	# 2. Iterative Process
	# 	# Velocity Update

	# 	self.v_dot_hat = self.g * np.array([0, 0, 1]) + self.R@np.array([0, 0, 1])*T + self.d_hat + As * (self.v_hat - self.v)
	# 	self.v_hat += (self.v_dot_hat*h)
	# 	#print(self.v_hat)
	# 	#print("v_hat:",self.v_hat)
	# 	#print((self.v))
	# 	#print(type(self.v))
		
	# 	# Disturbance Estimation
		
	# 	d_new = -(As * np.exp(As * h)* (np.linalg.inv(np.exp(As * h) - np.eye(3)))  @ (self.v_hat - self.v))
	# 	print(d_new)
	# 	# Low-pass Filter
	# 	self.d_hat = (alpha * self.d_hat + (1 - alpha) * d_new)

	# 	#print(self.d_hat)


	# 	z_d = f_d / np.linalg.norm(f_d)
	# 	e3 = np.array([0, 0, 1])
	# 	n = np.cross(e3, z_d)     # Compute rotation axis
	# 	rho = np.arcsin(np.linalg.norm(n))  # Compute rotation angle
		
	# 	n_normalized = n / np.linalg.norm(n)
	# 	rotation_vector = rho * n_normalized
	# 	REB = Rotation.from_rotvec(rotation_vector).as_matrix()  # Get the matrix representation of the rotation

	# 	RAE = Rotation.from_euler('z', yaw_d).as_matrix()

	# 	Rd = RAE @ REB

	# 	# attitude control
	# 	K_Ptau = 200
	# 	K_Dtau = 20
		
	# 	# Compute Re using given formula
	# 	Re = Rd.T @ self.R
		
	# 	R_diff = Re-Re.T
	# 	S_inv = np.array([-R_diff[1,2], R_diff[0,2], -R_diff[0,1]])
		
	# 	# Desired angular velocities are 0 (as given in slide)
	# 	omega_d = np.array([0, 0, 0])
	# 	alpha_d = np.array([0, 0, 0])  # Assumed to be zero unless specified

		
	# 	# Control input for attitude control using the formula from the slide
	# 	alpha = -K_Ptau * S_inv - K_Dtau * (self.omega - omega_d) + alpha_d
		
	# 	J_alpha = np.dot(self.J, alpha)
	# 	omega_cross_omega = np.cross(np.dot(self.J,self.omega), self.omega)
	# 	tau = J_alpha - omega_cross_omega
		

	# 	u = self.B_inv @ np.concatenate((np.array([T]),tau))
		
	# 	return u.flatten()

	def adaptive_control(self, p_d, v_d, a_d, yaw_d):
		'''
		Problem B-3: Based on lecture 4, implement adaptive control methods.
		For integral control, this function should be same as cascaded_control, 
		with an extra I gain in the position control loop.
		Hint for integral control: you can use self.d_hat to accumlate/integrate the position error.
		'''
		
		# position control
		K_P = 2.5
		K_D = 2.5
		K_I = 0.002

		e_p = self.p - p_d
		e_v = self.v - v_d
		self.d_hat += np.array(e_p)
		
		f_d = self.g*np.array([0,0,1]) - K_P * e_p - K_D * e_v + a_d - (K_I * self.d_hat)
		z = self.R @ np.array([0, 0, 1])
		T = np.dot(f_d.T, z) * self.m
		
		z_d = f_d / np.linalg.norm(f_d)
		e3 = np.array([0, 0, 1])
		n = np.cross(e3, z_d)     # Compute rotation axis
		rho = np.arcsin(np.linalg.norm(n))  # Compute rotation angle
		
		n_normalized = n / np.linalg.norm(n)
		rotation_vector = rho * n_normalized
		REB = Rotation.from_rotvec(rotation_vector).as_matrix()  # Get the matrix representation of the rotation

		RAE = Rotation.from_euler('z', yaw_d).as_matrix()

		Rd = RAE @ REB

		# attitude control
		K_Ptau = 200
		K_Dtau = 20
		
		# Compute Re using given formula
		Re = Rd.T @ self.R
		
		R_diff = Re-Re.T
		S_inv = np.array([-R_diff[1,2], R_diff[0,2], -R_diff[0,1]])
		
		# Desired angular velocities are 0 (as given in slide)
		omega_d = np.array([0, 0, 0])
		alpha_d = np.array([0, 0, 0])  # Assumed to be zero unless specified

		
		# Control input for attitude control using the formula from the slide
		alpha = -K_Ptau * S_inv - K_Dtau * (self.omega - omega_d) + alpha_d
		
		J_alpha = np.dot(self.J, alpha)
		omega_cross_omega = np.cross(np.dot(self.J,self.omega), self.omega)
		tau = J_alpha - omega_cross_omega
		

		u = self.B_inv @ np.concatenate((np.array([T]),tau))
		
		return u.flatten()

def plot(time, pos, vel, control, euler_rpy, omega, pos_des):
	plt.figure(figsize=(20, 4))
	plt.subplot(1, 5, 1)
	colors = ['tab:blue', 'tab:orange', 'tab:green']
	names = ['x', 'y', 'z']
	for i in range(3):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 5, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y", "z"])
	plt.subplot(1, 5, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2", "3", "4"])
	plt.subplot(1, 5, 4)
	plt.plot(time, euler_rpy)
	plt.xlabel("time (s)")
	plt.legend(["roll (rad)", "pitch (rad)", "yaw (rad)"])
	plt.subplot(1, 5, 5)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.ylabel("angular rate (rad/s)")
	plt.legend(["x", "y", "z"])
	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	robot = Quadrotor()
	total_time = 3 * np.pi
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 3))
	pos_des = np.zeros((total_step, 3))
	vel = np.zeros((total_step, 3))
	control = np.zeros((total_step, 4))
	control[0, :] = robot.u
	quat = np.zeros((total_step, 4))
	quat[0, :] = robot.q
	euler_rpy = np.zeros((total_step, 3))
	omega = np.zeros((total_step, 3))

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question

	'''
	Problem B-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.006, 0.008, 0.010, 0.012]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('quaternion:', robot.q)
				print('omega:', robot.omega)

	'''
	Problem B-2: cascaded tracking control
	'''
	robot.reset()
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		t = robot.t
		p_d = np.array([1,1,1])
		#p_d = np.array([np.sin(2*t),np.cos(2*t)-1,0.5*t])
		v_d = np.array([0,0,0])
		#v_d = np.array([2*np.cos(2*t),-2*np.sin(2*t),0.5])
		a_d = np.array([0,0,0])
		#a_d = np.array([-4*np.sin(2*t),-4*np.cos(2*t),0])
		yaw_d = 0#(t/total_time)*(np.pi/3)
		u = robot.cascaded_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
		omega[robot.step] = robot.omega
	if question == 2:
		rmse_x = compute_rmse(pos_des[:, 0], pos[:, 0])
		rmse_y = compute_rmse(pos_des[:, 1], pos[:, 1])
		rmse_z = compute_rmse(pos_des[:, 2], pos[:, 2])
		rmse_position = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)
		print("RMSE for x:", rmse_x)
		print("RMSE for y:", rmse_y)
		print("RMSE for z:", rmse_z)
		print("Combined RMSE for position:", rmse_position)
		control_energy = sum(np.linalg.norm(u)**2 for u in control) * robot.dt
		average_control_energy = control_energy / (len(control) * robot.dt)
		print("Average Control Energy:", average_control_energy)
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)

	'''
	Problem B-3: integral and adaptive control
	'''
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		t = robot.t
		robot.d = np.array([0.5,np.sin(t),np.cos(np.sqrt(2)*t)])#np.array([0.5, 0.5, 1])
		#p_d = np.array([1,1,1])
		p_d = np.array([np.sin(2*t),np.cos(2*t)-1,0.5*t])
		#v_d = np.array([0,0,0])
		v_d = np.array([2*np.cos(2*t),-2*np.sin(2*t),0.5])
		#a_d = np.array([0,0,0])
		a_d = np.array([-4*np.sin(2*t),-4*np.cos(2*t),0])
		yaw_d = 0.
		u = robot.adaptive_control(p_d, v_d, a_d, yaw_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		quat[robot.step] = robot.q
		euler_rpy[robot.step] = robot.euler_rpy
		omega[robot.step] = robot.omega
	if question == 3:
		
		directions = ['X', 'Y', 'Z']
		rise_times = []
		max_overshoots = []

		for i, direction in enumerate(directions):
			final_value = pos_des[-1, i]
			start_value = pos[0, i]
			rise_start = start_value + 0.1 * (final_value - start_value)
			rise_end = start_value + 0.9 * (final_value - start_value)

			indices = np.where((pos[:, i] >= rise_start) & (pos[:, i] <= rise_end))[0]
			if indices.size:
				rise_time = time[indices[-1]] - time[indices[0]]
			else:
				rise_time = 0
			rise_times.append(rise_time)

			# Calculate Maximum Overshoot for each direction
			overshoot = np.max(pos[:, i] - final_value)  # For current direction
			max_overshoots.append(overshoot)

		# Calculate Position RMSE for all directions
		pos_error = pos - pos_des
		rmse = np.sqrt(np.mean(pos_error**2, axis=0))

		# Display the results
		for i, direction in enumerate(directions):
			print(f"Rise Time ({direction}-direction): {rise_times[i]}")
			print(f"Position RMSE ({direction}-direction): {rmse[i]}")
			print(f"Maximum Overshoot ({direction}-direction): {max_overshoots[i]}")
		plot(time, pos, vel, control, euler_rpy, omega, pos_des)

	'''
	Animation using meshcat
	'''
	vis = meshcat.Visualizer()
	vis.open()

	vis["/Cameras/default"].set_transform(
		tf.translation_matrix([0,0,0]).dot(
		tf.euler_matrix(0,np.radians(-30),-np.pi/2)))

	vis["/Cameras/default/rotated/<object>"].set_transform(
		tf.translation_matrix([1,0,0]))

	vis["Quadrotor"].set_object(geometry.StlMeshGeometry.from_file('./crazyflie2.stl'))
	
	vertices = np.array([[0,0.5],[0,0],[0,0]]).astype(np.float32)
	vis["lines_segments"].set_object(geometry.Line(geometry.PointsGeometry(vertices), \
									 geometry.MeshBasicMaterial(color=0xff0000,linewidth=100.)))
	
	while True:
		for i in range(total_step):
			vis["Quadrotor"].set_transform(
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))
			vis["lines_segments"].set_transform(
				tf.translation_matrix(pos[i]).dot(tf.quaternion_matrix(quat[i])))				
			sleep(robot.dt)