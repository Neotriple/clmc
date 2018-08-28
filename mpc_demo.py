import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn # For nice looking plots.
import numpy as np
from pinocchio.utils import zero
import autograd.numpy as agnp
from autograd import grad
from autograd import jacobian

import pyipopt

from py_teststand_contact_learning.robot import build_teststand_robot
from py_teststand_contact_learning.dynamics import TeststandDynamics, ForceRecorderDynamics, plot_traj

np.set_printoptions(precision=2, suppress=True, linewidth=140)

class LearnedContact(object):
    def reset_contact(self, p):
        pass

    def __call__(self, dyn, x, u, dt):
        obs = dyn.observation(x, u)

        # TODO: Network force prediction goes here.
        fx, fy, fz = 0., 0., 1.

        return np.reshape([fx, fy, fz, 0., 0., 0.],(6, 1))


if __name__ == "__main__":


    robot, x = build_teststand_robot()
    x0 = zero(8)
    x0[:6] = x

    contact_model = {
        'type': 'SpringDamper',
        'parameters': [1e4, 80, 0., False], # Not using a friction cone.
        'terrain': lambda x, y: 0.,
    }

    timeSteps = 3
    dt = 0.001


    #PyIpopt stuff

    nvar = 9
    x_L = ones((nvar), dtype=float_) * 1.0
    x_U = ones((nvar), dtype=float_) * 5.0

    ncon = 9

    g_L = array([25.0, 40.0])
    g_U = array([2.0*pow(10.0, 19), 40.0]) 


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
        

    nnzj = 8  
    def eval_autojac_g(x, flag, user_data = None):
        if flag:
            return (array([0, 0, 0, 0, 1, 1, 1, 1]), 
                array([0, 1, 2, 3, 0, 1, 2, 3]))
        else:
            jac_g = jacobian(eval_g2)
            return jac_g(x)


    def apply_new(x):
        print("Here")
        return True
        
    nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
    nlp2 = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_autograd_f, eval_g2, eval_autojac_g)

    x0 = np.array([1.0, 5.0, 5.0, 1.0])

    print "Going to call solve for normal test case:"
    print x0
    x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
    nlp.close()

    #Dynamics Stuff
    dyn = TeststandDynamics(robot, contact_model, 100, dt=1e-3)
    dyn = ForceRecorderDynamics(dyn, 100)

    traj = dyn.integrate_policy(x0.reshape(-1), lambda t, x: np.array([-0.5, 0.5]), 100)



