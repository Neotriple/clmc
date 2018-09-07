#Constant Simulation Test

from pinocchio.robot_wrapper import RobotWrapper
from os.path import join
import pinocchio as se3
from pinocchio.utils import *
import numpy as np
import scipy as sp
import time

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper(URDF, [PKG])

robot.initDisplay(loadModel=True)

#Initial Set up & Variables
q = rand(robot.nq)
vq = zero(robot.nv)
aq = zero(robot.nv)
se3.forwardKinematics(robot.model, robot.data, q)
robot.display(q)
input_specified = None
dt = 0.001


while 1:
	if input_specified:
		pass
	else:
		tau = np.full((1,robot.nv), 10).T
		#tau = zero(robot.nv)

	aq = se3.aba(robot.model, robot.data, q, vq, tau)
	vq += aq * dt
	q = se3.integrate(robot.model, q, vq*dt)
	robot.display(q)
