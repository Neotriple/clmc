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

q = rand(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

robot.display(q)
time.sleep(2.5)

visualObj = robot.visual_model.geometryObjects[4]
visualName = visualObj.name
visualRef = robot.getViewerNodeName(visualObj, se3.GeometryType.VISUAL)

rgbt = [1.0, 1.0, 1.0, 1.0]
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)
qSphere = [0.5, .1, .2, 1.0, 0, 0, 0]
robot.viewer.gui.applyConfiguration("world/sphere", qSphere)
robot.viewer.gui.refresh()

qSphere = np.asmatrix(qSphere[:3])
qSphere = qSphere.T

#integrate
for i in range(1000):
	error = qSphere - robot.data.oMi[6].translation
	error = np.concatenate((error,np.matrix('0, 0, 0').T))
	#change q for every time step
	Jinv = np.linalg.pinv(robot.computeJacobians(q))
	qdot = Jinv*error
	robot.increment(q, qdot*.001)
	robot.display(q)
	if i == 0:
		print("first timestep")
		time.sleep(2.5)
	if i == 1:
		print("second timestep")
		time.sleep(2.5)