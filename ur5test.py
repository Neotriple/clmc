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

qEnd = [0.35, 0.25, 0.3]
qEnd = np.asmatrix(qEnd).T

qEnd2 = [0.35, 0.25, 0.0]
qEnd2 = np.asmatrix(qEnd2).T

#integrate
for i in range(5000):
	error = qSphere - robot.data.oMi[6].translation
	error = np.concatenate((error,np.matrix('0, 0, 0').T))
	#change q for every time step
	Jinv = np.linalg.pinv(robot.computeJacobians(q))
	qdot = Jinv*error
	robot.increment(q, qdot*.001)
	robot.display(q)

#integrate with ball in hand
for i in range(5000):
	error = qEnd - robot.data.oMi[6].translation
	error = np.concatenate((error,np.matrix('0, 0, 0').T))
	#change q for every time step
	Jinv = np.linalg.pinv(robot.computeJacobians(q))
	qdot = Jinv*error
	robot.increment(q, qdot*.001)
	robot.display(q)
	q2 = robot.data.oMi[6].translation
	q2 = np.concatenate((q2,np.matrix('1, 0, 0, 0').T)).T
	q2 = (np.asarray(q2).reshape(-1)).tolist()
	robot.viewer.gui.deleteNode("world/sphere", 1)
	robot.viewer.gui.addSphere("world/sphere", .1, rgbt)
	robot.viewer.gui.refresh()
	robot.viewer.gui.applyConfiguration("world/sphere", q2)
	robot.viewer.gui.refresh()

for i in range(5000):
	error = qEnd2 - robot.data.oMi[6].translation
	error = np.concatenate((error,np.matrix('0, 0, 0').T))
	#change q for every time step
	Jinv = np.linalg.pinv(robot.computeJacobians(q))
	qdot = Jinv*error
	robot.increment(q, qdot*.01)
	robot.display(q)
	q2 = robot.data.oMi[6].translation
	q2 = np.concatenate((q2,np.matrix('1, 0, 0, 0').T)).T
	q2 = (np.asarray(q2).reshape(-1)).tolist()
	robot.viewer.gui.deleteNode("world/sphere", 1)
	robot.viewer.gui.addSphere("world/sphere", .1, rgbt)
	robot.viewer.gui.refresh()
	robot.viewer.gui.applyConfiguration("world/sphere", q2)
	robot.viewer.gui.refresh()

