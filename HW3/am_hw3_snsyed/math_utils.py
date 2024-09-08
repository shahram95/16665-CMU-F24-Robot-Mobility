"""
Air Mobility Project- 16665
Author: Guanya Shi
guanyas@andrew.cmu.edu
"""

import numpy as np

def qmultiply(q1, q2):
	return np.concatenate((
		np.array([q1[0] * q2[0] - np.sum(q1[1:4] * q2[1:4])]), # w1w2
		q1[0] * q2[1:4] + q2[0] * q1[1:4] + np.cross(q1[1:4], q2[1:4])))

def qconjugate(q):
	return np.concatenate((q[0:1],-q[1:4]))

def qrotate(q, v):
	quat_v = np.concatenate((np.array([0]), v))
	return qmultiply(q, qmultiply(quat_v, qconjugate(q)))[1:]

def qexp(q):
	norm = np.linalg.norm(q[1:4])
	e = np.exp(q[0])
	result_w = e * np.cos(norm)
	if np.isclose(norm, 0):
		result_v = np.zeros(3)
	else:
		result_v = e * q[1:4] / norm * np.sin(norm)
	return np.concatenate((np.array([result_w]), result_v))

def qintegrate(q, v, dt):
	quat_v = np.concatenate((np.array([0]), v*dt/2))
	return qmultiply(q, qexp(quat_v))		

def qstandardize(q):
	if q[0] < 0:
		q *= -1
	return q / np.linalg.norm(q)

def qtoR(q):
	q0 = q[0]
	q1 = q[1]
	q2 = q[2]
	q3 = q[3]
     
    # First row of the rotation matrix
	r00 = 2 * (q0 * q0 + q1 * q1) - 1
	r01 = 2 * (q1 * q2 - q0 * q3)
	r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
	r10 = 2 * (q1 * q2 + q0 * q3)
	r11 = 2 * (q0 * q0 + q2 * q2) - 1
	r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
	r20 = 2 * (q1 * q3 - q0 * q2)
	r21 = 2 * (q2 * q3 + q0 * q1)
	r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
	rot_matrix = np.array([[r00, r01, r02], \
                           [r10, r11, r12], \
                           [r20, r21, r22]])
                            
	return rot_matrix

def vee(R):
	return np.array([R[2,1], R[0,2], R[1,0]])