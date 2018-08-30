import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn # For nice looking plots.
import numpy as np
import autograd.numpy as agnp
import pinocchio as se3
from autograd import grad
from autograd import jacobian
from os.path import join
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper
import time

import pyipopt

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

    robot.display(q)

    #PyIpopt stuff
    timeSteps = 3
    dt = 0.001
    nvar = robot.nv*timeSteps
    x_L = np.ones((nvar), dtype=np.float_) * 1.0
    x_U = np.ones((nvar), dtype=np.float_) * 50.0

    q_ref1 = np.array([-1.49411011e-02,-1.32003896e+00,2.09324591e+00,-7.73206945e-01,-1.49411010e-02,-2.32376275e-09])

    q_ref1 = np.asmatrix(q_ref1).T

    qRef = np.empty([timeSteps, 1, robot.nq, 1])

    temp = 0
    while (temp < timeSteps):
        qRef[temp, 0] = q_ref1
        temp += 1

    ncon = 1

    g_L = np.array([0.01])
    g_U = np.array([2.0*pow(10.0, 19)])
    #When starting off from 0 initial position & 0 initial velocity
    q0 = np.zeros(robot.nq)
    vq0 = np.zeros(robot.nv)
    aq0 = np.zeros(robot.nv)

    q0 = np.asmatrix(q0).T
    vq0 = np.asmatrix(vq0).T
    aq0 = np.asmatrix(aq0).T

    def eval_f(tau, user_data = None):
        assert len(tau) == nvar
        tempRobot = robot
        vectorNorm = agnp.linalg.norm(tau)

        # tauNorm = 0
        # for tk in tau:
        #     tauNorm += tk**2

        # tauNorm = agnp.sqrt(tauNorm)

        q = q0
        vq = vq0
        aq = aq0
        qNorm = agnp.empty([timeSteps, 1, robot.nq, 1])
        t = 0
        while (t < timeSteps):
            tau_k = tau[t*timeSteps:(t*timeSteps)+robot.nv]
            tau_k = np.asmatrix(tau_k).T
            b_k = se3.rnea(tempRobot.model, tempRobot.data, q, vq, aq)
            M_k = se3.crba(tempRobot.model, tempRobot.data, q)
            aq = np.linalg.inv(M_k)*(tau_k - b_k)
            vq += dt*aq
            q = se3.integrate(tempRobot.model, q, vq*dt)
            qNorm[t, 0] = q
            t += 1

        #qNorm = np.linalg.norm(qRef - qNorm)
        qNorm = agnp.linalg.norm(qNorm-qRef)
        #return vectorNorm + qNorm
        return vectorNorm + qNorm


    def eval_autograd_f(tau):
        grad_eval_f = grad(eval_f)
        return grad_eval_f(tau)

    def eval_grad_f_finiteDiff(tau):

        grad_f = np.empty([timeSteps*robot.nv])
        diffStep = dt/10

        i = 0
        while(i < len(tau)):
            tauTemp = tau
            #Calculate Upper Limit
            tauTemp[i] = tau[i] + diffStep
            upperLim = eval_f(tauTemp)
            tauTemp[i] = tau[i] - diffStep
            bottomLim = eval_f(tauTemp)
            deriv = (upperLim - bottomLim)/diffStep
            grad_f[i] = deriv
            i += 1

        return grad_f
        
        
    def eval_g2(tau, user_data= None):
        assert len(tau) == nvar

        C_tau = agnp.ones(robot.nv*timeSteps)

        constraints = C_tau*tau
        return constraints

    nnzj = robot.nv*timeSteps
    def eval_autojac_g(tau, flag, user_data = None):
        if flag:
            array1 = agnp.zeros(nvar)
            array2 = agnp.linspace(0, robot.nv*timeSteps - 1, robot.nv*timeSteps)
            return (array1, array2)
        else:
            jac_g = jacobian(eval_g2)
            return jac_g(tau)


    def apply_new(tau):
        print("Here")
        return True
    nnzh = 0

    nlp2 = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f_finiteDiff, eval_g2, eval_autojac_g)

    x0 = agnp.zeros(nvar)

    print "Going to call solve for normal test case:"
    print x0
    traj, zl, zu, constraint_multipliers, obj, status = nlp2.solve(x0)
    nlp2.close()


    print
    #print "Solution of the primal variables, x"
    #print_variable("x", x2)
    #print
    print "Objective value"
    print "f(x*) =", obj


    q = zero(robot.nq)
    vq = zero(robot.nv)
    t = 0
    while (t < timeSteps):
        tau_k = traj[t*timeSteps:(t*timeSteps)+robot.nv]
        tau_k = np.asmatrix(tau_k).T
        aq = se3.aba(robot.model, robot.data, q, vq, tau_k)
        vq += aq * dt
        q = se3.integrate(robot.model, q, vq*dt)
        robot.display(q)
        t += 1
        time.sleep(2.5)

