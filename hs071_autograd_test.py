#!/usr/bin/python

# Author: Eric Xu. Washington University
#  The same model as Ipopt/examples/hs071

import pyipopt
from numpy import *
import autograd.numpy as np
from autograd import grad
from autograd import jacobian

nvar = 4
x_L = ones((nvar), dtype=float_) * 1.0
x_U = ones((nvar), dtype=float_) * 5.0

ncon = 2

g_L = array([25.0, 40.0])
g_U = array([2.0*pow(10.0, 19), 40.0]) 



def eval_f(x, user_data = None):
	assert len(x) == 4
	return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

def eval_grad_f(x, user_data = None):
	assert len(x) == 4
	grad_f = array([
  		x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]) , 
  		x[0] * x[3],
  		x[0] * x[3] + 1.0,
  		x[0] * (x[0] + x[1] + x[2])
  		], float_)
	return grad_f;

def eval_autograd_f(x):
	grad_eval_f = grad(eval_f)
	return grad_eval_f(x)
	
def eval_g(x, user_data= None):
	assert len(x) == 4
	return array([
		x[0] * x[1] * x[2] * x[3], 
		x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
	], float_)

def eval_g2(x, user_data= None):
	assert len(x) == 4
	return np.array([
		x[0] * x[1] * x[2] * x[3], 
		x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
	], float_)

nnzj = 8
def eval_jac_g(x, flag, user_data = None):
	if flag:
		return (array([0, 0, 0, 0, 1, 1, 1, 1]), 
			array([0, 1, 2, 3, 0, 1, 2, 3]))
	else:
		assert len(x) == 4
		return array([ x[1]*x[2]*x[3], 
					x[0]*x[2]*x[3], 
					x[0]*x[1]*x[3], 
					x[0]*x[1]*x[2],
					2.0*x[0], 
					2.0*x[1], 
					2.0*x[2], 
					2.0*x[3] ])
		
def eval_autojac_g(x, flag, user_data = None):
	if flag:
		return (array([0, 0, 0, 0, 1, 1, 1, 1]), 
			array([0, 1, 2, 3, 0, 1, 2, 3]))
	else:
		jac_g = jacobian(eval_g2)
		return jac_g(x)

nnzh = 0
def eval_h(x, lagrange, obj_factor, flag, user_data = None):
	print("Here - eval H")
	if flag:
		hrow = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
		hcol = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
		return (array(hcol), array(hrow))
	else:
		values = zeros((10), float_)
		values[0] = obj_factor * (2*x[3])
		values[1] = obj_factor * (x[3])
		values[2] = 0
		values[3] = obj_factor * (x[3])
		values[4] = 0
		values[5] = 0
		values[6] = obj_factor * (2*x[0] + x[1] + x[2])
		values[7] = obj_factor * (x[0])
		values[8] = obj_factor * (x[0])
		values[9] = 0
		values[1] += lagrange[0] * (x[2] * x[3])

		values[3] += lagrange[0] * (x[1] * x[3])
		values[4] += lagrange[0] * (x[0] * x[3])

		values[6] += lagrange[0] * (x[1] * x[2])
		values[7] += lagrange[0] * (x[0] * x[2])
		values[8] += lagrange[0] * (x[0] * x[1])
		values[0] += lagrange[1] * 2
		values[2] += lagrange[1] * 2
		values[5] += lagrange[1] * 2
		values[9] += lagrange[1] * 2
		return values

def apply_new(x):
	print("Here")
	return True
	
nlp = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g)
nlp2 = pyipopt.create(nvar, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_autograd_f, eval_g2, eval_autojac_g)

x0 = np.array([1.0, 5.0, 5.0, 1.0])

print "Going to call solve for normal test case:"
print x0
x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
# import pdb; pdb.set_trace()
nlp.close()

def print_variable(variable_name, value):
  for i in xrange(len(value)):
    print variable_name + "["+str(i)+"] =", value[i]

print
#print "Solution of the primal variables, x"
#print_variable("x", x)
#print
print "Objective value"
print "f(x*) =", obj
print
print
print
print "Going to call solve for autograd test case:"
print x0
x2, zl2, zu2, constraint_multipliers2, obj2, status2 = nlp2.solve(x0)
# import pdb; pdb.set_trace()
nlp2.close()

def print_variable(variable_name, value):
  for i in xrange(len(value)):
    print variable_name + "["+str(i)+"] =", value[i]

print
#print "Solution of the primal variables, x"
#print_variable("x", x2)
#print
print "Objective value"
print "f(x*) =", obj2
