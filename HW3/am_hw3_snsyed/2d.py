"""
Air Mobility Project- 16665
Author: Guanya Shi
guanyas@andrew.cmu.edu
"""

import numpy as np
import meshcat
import meshcat.geometry as geometry
import meshcat.transformations as tf
import matplotlib.pyplot as plt
from time import sleep
from control import lqr
import argparse

def compute_rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual)**2))

class Quadrotor_2D():
	def __init__(self):
		# parameters
		self.m = 0.027 # kg
		self.J = 8.571710e-5 # inertia
		self.arm = 0.0325 # arm length
		self.g = 9.81 # gravity

		# control actuation matrix
		self.B = np.array([[1, 1],
						   [-self.arm, self.arm]])
		self.B_inv = np.linalg.inv(self.B)

		# noise level
		self.sigma_t = 0.25 # for translational dynamics
		self.sigma_r = 0.25 # for rotational dynamics

		# initial state
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.theta = 0.
		self.omega = 0.

		# initial control (hovering)
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])

		# control limit for each rotor (N)
		self.umin = 0.
		self.umax = 0.024 * self.g

		# total time and discretizaiton step
		self.dt = 0.01
		self.step = 0
		self.t = 0.

	def reset(self):
		self.sigma_t = 0.25
		self.sigma_r = 0.25
		self.p = np.array([0., 0])
		self.v = np.array([0., 0])
		self.theta = 0#np.pi/3.
		self.omega = 0.
		self.u = np.array([self.m*self.g/2, self.m*self.g/2])
		self.step = 0
		self.t = 0.

	def dynamics(self, u):
		'''
		Problem A-1: Based on lecture 2, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		self.u is the control input (two rotor forces).
		Hint: first convert self.u to total thrust and torque using the control actuation matrix.
		'''
		u = np.clip(u, self.umin, self.umax)
		self.u = np.clip(u, self.umin, self.umax)

		# Convert control inputs to total thrust and torque
		total_thrust = np.sum(self.u)
		torque = (self.u[1] - self.u[0]) * self.arm
		
		pdot = self.v
		vdot = np.array([0, -self.g]) + total_thrust/self.m * np.array([-np.sin(self.theta), np.cos(self.theta)])
		thetadot = self.omega
		omegadot = torque/self.J

		self.p += self.dt * pdot
		self.v += self.dt * vdot + self.dt * self.sigma_t * np.random.normal(size=2)
		self.theta += self.dt * thetadot
		self.omega += self.dt * omegadot + self.dt * self.sigma_r * np.random.normal()

		self.t += self.dt
		self.step += 1

	def cascaded_control(self, p_d, v_d, a_d, omega_d=0., tau_d=0.):
		'''
		Problem A-2 and A-4: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a cascaded controller to track a trajectory (p_d, v_d, a_d).
		omega_d=0, tau_d=0 except for Problem A-5.
		Hint for gain tuning: position control gain is smaller (1-10);
		Attitude control gain is bigger (10-200).
		'''
		# position control
		#K_P = np.array([2.5,2.5])
		#K_D = np.array([2.5,2.5])

		#K_P = np.array([1.75,1.75])
		#K_D = np.array([2.5,2.5])

		K_P = np.array([2.5,2.5])
		K_D = np.array([2.5,2.5])

		# attitude control
		K_Ptau = 156
		K_Dtau = 10

		# Step 1: Position Control to compute the desired force f_d
		f_d = self.g * np.array([0., 1.]) + a_d - K_P * (self.p - p_d) - K_D * (self.v- v_d)

		# Step 2: Convert f_d to desired thrust T and desired attitude theta_d
		T = self.m * np.dot(f_d, np.array([-np.sin(self.theta), np.cos(self.theta)]))
		theta_d = -np.arctan2(f_d[0], f_d[1])
		
		# Step 3: Attitude control to get desired torque tau
		tau = self.J * (-K_Ptau * (self.theta - theta_d) - K_Dtau * (self.omega - omega_d) + tau_d) 
		
		# Convert total thrust T and torque tau to individual rotor forces T1 and T2
		control_input = np.array([[1, 1], [-self.arm, self.arm]])
		u = np.linalg.inv(control_input) @ np.array([T, tau])
		
		return u

	def linear_control(self, p_d):
		'''
		Problem A-3: Based on lecture 3, complete the following codes.
		Please only complete the "..." parts. Don't change other codes.
		Your goal is to develop a LQR control based on the linearized model around the hovering condition.
		Hint: use the lqr function in the control library.
		'''

		A = np.array([
			[0, 0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, -self.g, 0],
			[0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0, 0]
		])
		B = np.array([
			[0, 0],
			[0, 0],
			[0, 0],
			[1, 0],
			[0, 0],
			[0, 1]
		])
		Q = np.array([
			[10, 0, 0, 0, 0, 0],
			[0, 10, 0, 0, 0, 0],
			[0, 0, 10, 0, 0, 0],
			[0, 0, 0, 10, 0, 0],
			[0, 0, 0, 0, 10, 0],
			[0, 0, 0, 0, 0, 10]
		])
		R = np.array([
			[0.1,0],
			[0,0.1]
		])
		K,_,_ = lqr(A,B,Q,R)
		position_error = self.p - np.array(p_d)
		
		state_error = np.vstack([position_error[0], position_error[1], self.v[0],self.v[1], self.theta, self.omega])
		t_tau = -np.dot(K, state_error).flatten()
		T,tau = self.m*(t_tau[0]+self.g), t_tau[1]*self.J
		control_input = np.array([[1, 1], [-self.arm, self.arm]])
		u = np.linalg.inv(control_input) @ np.array([T, tau])

		return u

def plot(time, pos, vel, control, theta, omega, pos_des):
	plt.figure(figsize=(16, 4))
	#plt.title("Cascaded Control to move from [0,0] to [1,1]. KP = 1.75, KD = 2.5, KPtau=150, KDtau=25")
	plt.subplot(1, 4, 1)
	colors = ['tab:blue', 'tab:orange']
	names = ['x', 'y']
	for i in range(2):
		plt.plot(time, pos[:,i], color=colors[i], label=names[i]+" actual")
		plt.plot(time, pos_des[:,i], '--', color=colors[i], label=names[i]+" desired")
	plt.xlabel("time (s)")
	plt.ylabel("pos (m)")
	plt.legend()
	plt.subplot(1, 4, 2)
	plt.plot(time, vel)
	plt.xlabel("time (s)")
	plt.ylabel("vel (m/s)")
	plt.legend(["x", "y"])
	plt.subplot(1, 4, 3)
	plt.plot(time, control)
	plt.xlabel("time (s)")
	plt.ylabel("control (N)")
	plt.legend(["1", "2"])
	plt.subplot(1, 4, 4)
	plt.plot(time, theta)
	plt.plot(time, omega)
	plt.xlabel("time (s)")
	plt.legend(["theta (rad)", "omega (rad/s)"])
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	robot = Quadrotor_2D()
	total_time = 2 * np.pi
	total_step = int(total_time/robot.dt+1)
	time = np.linspace(0, total_time, total_step)
	pos = np.zeros((total_step, 2))
	pos_des = np.zeros((total_step, 2))
	vel = np.zeros((total_step, 2))
	control = np.zeros((total_step, 2))
	control[0,:] = robot.u
	theta = np.zeros(total_step)
	omega = np.zeros(total_step)

	parser = argparse.ArgumentParser()
	parser.add_argument('question', type=int)
	question = parser.parse_args().question
	
	'''
	Problem A-1: system modeling
	'''
	if question == 1:
		robot.sigma_r = 0.
		robot.sigma_t = 0.
		for i in range(21):
			u = np.array([0.019, 0.023]) * 9.81
			robot.dynamics(u)
			if i % 10 == 0:
				print('************************')
				print('pos:', robot.p)
				print('vel:', robot.v)
				print('theta:', robot.theta)
				print('omega:', robot.omega)

	'''
	Problem A-2: cascaded setpoint control
	Complete p_d, v_d, and a_d
	'''
	robot.reset()
	while True:
		if question != 2 or robot.step >= total_step-1:
			break
		p_d = [1,0]
		v_d = [0,0]
		a_d = [0,0]
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 2:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)

		labels = ["x-position", "y-position"]

		for idx, label in enumerate(labels):
			# 1. Rise Time (90%)
			final_value = pos[-1, idx]
			start_value = pos[0, idx]
			time_10_percent = next(t for t, p in enumerate(pos[:, idx]) if p >= start_value + 0.1 * (p_d[idx] - start_value)) * robot.dt
			time_90_percent = next(t for t, p in enumerate(pos[:, idx]) if p >= start_value + 0.9 * (p_d[idx] - start_value)) * robot.dt
			rise_time_90 = time_90_percent - time_10_percent

			# 2. Maximum Overshoot
			overshoot = (max(pos[:, idx]) - p_d[idx]) / p_d[idx] * 100
			
			print(f"For {label}:")
			print("Rise Time (90%):", rise_time_90)
			print("Maximum Overshoot (%):", overshoot)

		# 3. Average Control Energy
		control_energy = sum(np.linalg.norm(u)**2 for u in control) * robot.dt
		average_control_energy = control_energy / (len(control) * robot.dt)
		print("Average Control Energy:", average_control_energy)

	'''
	Problem A-3: linear setpoint control
	Complete p_d
	'''
	robot.reset()
	while True:
		if question != 3 or robot.step >= total_step-1:
			break
		p_d = [1,0]
		u = robot.linear_control(p_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 3:
		pos_des[0,:] = p_d
		plot(time, pos, vel, control, theta, omega, pos_des)

		labels = ["x-position", "y-position"]

		for idx, label in enumerate(labels):
			# 1. Rise Time (90%)
			final_value = pos[-1, idx]
			start_value = pos[0, idx]
			time_10_percent = next(t for t, p in enumerate(pos[:, idx]) if p >= start_value + 0.1 * (p_d[idx] - start_value)) * robot.dt
			time_90_percent = next(t for t, p in enumerate(pos[:, idx]) if p >= start_value + 0.9 * (p_d[idx] - start_value)) * robot.dt
			rise_time_90 = time_90_percent - time_10_percent

			# 2. Maximum Overshoot
			overshoot = (max(pos[:, idx]) - p_d[idx]) / p_d[idx] * 100
			
			print(f"For {label}:")
			print("Rise Time (90%):", rise_time_90)
			print("Maximum Overshoot (%):", overshoot)

		# 3. Average Control Energy
		control_energy = sum(np.linalg.norm(u)**2 for u in control) * robot.dt
		average_control_energy = control_energy / (len(control) * robot.dt)
		print("Average Control Energy:", average_control_energy)

	'''
	Problem A-4: cascaded tracking control
	Complete p_d, v_d, and a_d
	'''
	robot.reset()
	while True:
		if question != 4 or robot.step >= total_step-1:
			break
		t = robot.t
		p_d = [np.sin(robot.t), 0.5 * np.cos(2*robot.t + np.pi/2)]
		v_d = [0,0]#[np.cos(robot.t), -np.sin(2*robot.t + np.pi/2)]
		a_d = [0,0] # [-np.sin(robot.t), -2*np.cos(2*robot.t + np.pi/2)]
		u = robot.cascaded_control(p_d, v_d, a_d)
		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_d
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 4:
		rmse_x = compute_rmse(pos_des[:, 0], pos[:, 0])
		rmse_y = compute_rmse(pos_des[:, 1], pos[:, 1])
		rmse_position = np.sqrt(rmse_x**2 + rmse_y**2)
		print("RMSE for x:", rmse_x)
		print("RMSE for y:", rmse_y)
		print("Combined RMSE for position:", rmse_position)
		plot(time, pos, vel, control, theta, omega, pos_des)

	'''
	Problem A-5: trajectory generation and differential flatness
	Design trajectory and tracking controllers here.
	'''
	robot.reset()
	while True:
		if question != 5 or robot.step >= total_step-1:
			break
		t = robot.t 
		pos_x_cof = np.array([0.00000000e+00, -1.18423789e-15, 1.04546001e-15, -4.47173163e-16, 5.40123457e-02, -2.88065844e-02, 6.00137174e-03, -5.71559214e-04, 2.08380963e-05])
		
		#pos_x_cof = np.array([0.00000000e+00, -1.42108547e-14,  0.00000000e+00,  0.00000000e+00, 4.37500000e+00, -7.00000000e+00,  4.37500000e+00, -1.25000000e+00, 1.36718750e-01])
		vel_x_cof = np.polyder(pos_x_cof[::-1])[::-1]
		acc_x_cof = np.polyder(vel_x_cof[::-1])[::-1]
		j_x_cof = np.polyder(acc_x_cof[::-1])[::-1]
		s_x_cof = np.polyder(j_x_cof[::-1])[::-1]
		
		pos_y_cof = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
		vel_y_cof = np.polyder(pos_y_cof[::-1])[::-1]
		acc_y_cof = np.polyder(vel_y_cof[::-1])[::-1]
		j_y_cof = np.polyder(acc_y_cof[::-1])[::-1]
		s_y_cof = np.polyder(j_y_cof[::-1])[::-1]

		p_r = np.array([np.polyval(pos_x_cof[::-1], robot.t), 0]) #np.polyval(pos_y_cof[::-1], robot.t)])
		v_r = np.array([np.polyval(vel_x_cof[::-1], robot.t), 0]) #np.polyval(vel_y_cof[::-1], robot.t)])
		a_r = np.array([np.polyval(acc_x_cof[::-1], robot.t), 0]) #np.polyval(acc_y_cof[::-1], robot.t)])
		j_r = np.array([np.polyval(j_x_cof[::-1], robot.t), 0]) #np.polyval(j_y_cof[::-1], robot.t)])
		s_r = np.array([np.polyval(s_x_cof[::-1], robot.t), 0]) #np.polyval(s_y_cof[::-1], robot.t)])

		'''p_r = np.array([robot.t/total_time,0])
		v_r = np.array([1/total_time,0])
		a_r = np.array([0,0])
		j_r = np.array([0,0])
		s_r = np.array([0,0])'''
		'''
		differential flatness
		'''
		y_vec = np.array([-np.sin(robot.theta),np.cos(robot.theta)]).T
		x_vec = np.array([np.cos(robot.theta),np.sin(robot.theta)]).T
		T = (a_r - np.array([0, -9.81])).T @ y_vec
		omega_d = -(j_r.T @ x_vec)/T
		tau_d = -((s_r.T @ x_vec) + 2*(j_r.T@y_vec)*omega_d)/T 
		print(omega_d)
		print(tau_d)
		u = robot.cascaded_control(p_r, v_r, a_r, 0,0)#omega_d, tau_d)

		robot.dynamics(u)
		pos[robot.step,:] = robot.p
		pos_des[robot.step,:] = p_r
		vel[robot.step,:] = robot.v
		control[robot.step,:] = robot.u
		theta[robot.step] = robot.theta
		omega[robot.step] = robot.omega
	if question == 5:
		rmse_x = compute_rmse(pos_des[:, 0], pos[:, 0])
		rmse_y = compute_rmse(pos_des[:, 1], pos[:, 1])
		rmse_position = np.sqrt(rmse_x**2 + rmse_y**2)
		print("RMSE for x:", rmse_x)
		print("RMSE for y:", rmse_y)
		print("Combined RMSE for position:", rmse_position)
		plot(time, pos, vel, control, theta, omega, pos_des)

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
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))
			vis["lines_segments"].set_transform(
				tf.translation_matrix([pos[i,0], 0, pos[i,1]]).dot(tf.euler_matrix(0, theta[i], 0)))				
			sleep(robot.dt)