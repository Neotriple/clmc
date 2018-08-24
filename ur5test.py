from pinocchio.robot_wrapper import RobotWrapper
from os.path import join
import pinocchio as se3
from pinocchio.utils import *
import numpy as np
import scipy as sp

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'ur_description/urdf/ur5_gripper.urdf')
robot = RobotWrapper(URDF, [PKG])

robot.initDisplay(loadModel=True)

q = zero(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

visualObj = robot.visual_model.geometryObjects[4]
visualName = visualObj.name
visualRef = robot.getViewerNodeName(visualObj, se3.GeometryType.VISUAL)

rgbt = [1.0, 1.0, 1.0, 1.0]
robot.viewer.gui.addSphere("world/sphere", .1, rgbt)
qSphere = [0.5, .1, .2, 1.0, 0, 0, 0]
robot.viewer.gui.applyConfiguration("world/sphere", qSphere)
robot.viewer.gui.refresh()

q2 = np.matrix('1; 1; 1; 1; 1; 1')

for i in range(1000):
	error = robot.data.oMi[6].translation - qSphere[:3]
	Jinv = np.linalg.pinv(robot.computeJacobians(q))
	qdot = Jinv*error
	robot.increment(q, qdot*.001)
	robot.display(q)

#print(placement)

#placement = robot.data.oMi[idx].copy()

#print(placement)