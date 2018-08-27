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
    qv0 = zero(robot.nv)
    aq0 = zero(robot.nv)
    
    def eval_f(tau, user_data = None):
        assert len(tau) == robot.nv*timeSteps

        vectorNorm = np.linalg.norm(tau)

        t = 0
        while t < timeSteps:
            robot.

        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]


    def eval_autograd_f(x):
        grad_eval_f = grad(eval_f)
        return grad_eval_f(x)
        

    def eval_g2(x, user_data= None):
        assert len(x) == 4
        return np.array([
            x[0] * x[1] * x[2] * x[3], 
            x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
        ], float_)

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



