import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn # For nice looking plots.
import numpy as np
import autograd.numpy as agnp
from autograd import grad
from autograd import jacobian
from os.path import join
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

import pyipopt


np.set_printoptions(precision=2, suppress=True, linewidth=140)

if __name__ == "__main__":


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

    #PyIpopt stuff
    timeSteps = 3
    dt = 0.001
    nvar = robot.nv*timeSteps
    x_L = ones((nvar), dtype=float_) * 1.0
    x_U = ones((nvar), dtype=float_) * 50.0

    ncon = 9

    g_L = array([0.00])
    g_U = array([50*robot.nv]) 


    #When starting off from 0 initial position & 0 initial velocity
    q0 = zero(robot.nq)
    vq0 = zero(robot.nv)
    aq0 = zero(robot.nv)

    def eval_f(tau, user_data = None):
        assert len(tau) == robot.nv*timeSteps

        tempRobot = robot
        vectorNorm = np.linalg.norm(tau)
        q = q0
        vq = vq0
        aq = aq0
        t = 0
        qNorm = np.empty([timeSteps, 1, robot.nq, 1])
        while t < timeSteps:
            tau_k = tau[t*timeSteps:(t*timeSteps)+robot.nv]
            b_k = se3.rnea(tempRobot.model, tempRobot.data, q, vq, aq)
            M_k = se3.crba(tempRobot.model, tempRobot.data, q)
            aq = np.linalg.inv(M_k)*(tau_k - b_k)
            vq += dt*aq
            q = se3.integrate(robot.model, q, vq*dt)
            qNorm[t, 0] = q
            t += 1

        qNorm = np.linalg.norm(q_ref - qNorm)

        return vectorNorm + qNorm


    def eval_autograd_f(x):
        grad_eval_f = grad(eval_f)
        return grad_eval_f(x)
        
    def eval_g2(tau, user_data= None):
        assert len(tau) == robot.nv*timeSteps

        C_tau = agnp.ones(robot.nv*timeSteps)

        constraints = C_tau*tau
        return constraints

    nnzj = robot.nv*timeSteps
    def eval_autojac_g(tau, flag, user_data = None):
        if flag:
            return (np.linspace(0, robot.nv*timeSteps - 1, robot.nv*timeSteps))
        else:
            jac_g = jacobian(eval_g2)
            return jac_g(x)


    def apply_new(x):
        print("Here")
        return True
    
    nlp2 = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_autograd_f, eval_g2, eval_autojac_g)

    x0 = agnp.zeros(nvar)

    print "Going to call solve for normal test case:"
    print x0
    traj, zl, zu, constraint_multipliers, obj, status = nlp2.solve(x0)
    nlp2.close()


    q = zero(robot.nq)
    vq = zero.(robot.nv)
    while t < timeSteps:
        tau_k = traj[t*timeSteps:(t*timeSteps)+robot.nv]
        se3.aba(robot.model, robot.data, q, dq, tau_k)
        robot.display(q)


