#Romeo Test

from pinocchio.romeo_wrapper import RomeoWrapper
from os.path import join
import pinocchio as se3
from pinocchio.utils import *
import numpy as np
import scipy as sp
import time

PKG = '/opt/openrobots/share'
URDF = join(PKG, 'romeo_description/urdf/romeo.urdf')
robot = RomeoWrapper(URDF, [PKG])

robot.initDisplay(loadModel=True)

q = rand(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

robot.display(q)

q = rand(robot.nq)

se3.forwardKinematics(robot.model, robot.data, q)

robot.display(q)