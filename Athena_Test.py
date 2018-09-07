# Athena Test

from pinocchio.robot_wrapper import RobotWrapper
from os.path import join
import pinocchio as se3
from pinocchio.utils import *
import numpy as np
import scipy as sp
import time

PKG = '/Network/Servers/duerer/Volumes/duerer/paarth/Simulation/workspace/src/catkin/humanoids/humanoid_control/robot_properties'
URDF = join(PKG, 'robot_properties_hermes_full/urdf/hermes_full.urdf')
robot = RobotWrapper(URDF, [PKG])

robot.initDisplay(loadModel=True)

q = rand(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

robot.display(q)

q = rand(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

robot.display(q)