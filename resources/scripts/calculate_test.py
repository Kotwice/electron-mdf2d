
from numpy import linspace, zeros, meshgrid, diff, ones, arange, array, append, reshape, concatenate, stack, sqrt, split, array_split, sum, abs
from numpy.matlib import repmat
from matplotlib import pyplot as plt
from scipy.linalg import solve as linsolve
from scipy.optimize import fsolve
import multiprocessing as mp
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from numba import jit
import json, socketio, numpy, sys

from scipy.optimize import fsolve

def fig2json(x, y, t, u, v, p):

    data_u = [go.Contour(x = x['array'], y = y['array'], z = u[:, :, i], visible = False, showscale = False, name = 'u') for i in arange(t['node'])]
    data_v = [go.Contour(x = x['array'], y = y['array'], z = v[:, :, i], visible = False, showscale = False, name = 'v') for i in arange(t['node'])]
    data_p = [go.Contour(x = x['array'], y = y['array'], z = p[:, :, i], visible = False, showscale = False, name = 'p') for i in arange(t['node'])]

    data_u[0]['visible'] = True
    data_v[0]['visible'] = True
    data_p[0]['visible'] = True

    figure = make_subplots(rows = 3, cols = 1)

    for i in arange(t['node']):
        figure.append_trace(data_u[i], row = 1, col = 1)
        figure.append_trace(data_v[i], row = 2, col = 1)
        figure.append_trace(data_p[i], row = 3, col = 1)

    steps = [
        dict(
            method = 'update',
            args = [dict(visible = [False] * len(figure.data))],
            label = round(t['array'][i], 4),
        ) 
        for i in arange(t['node'])
    ]

    index = arange(len(figure.data)).reshape((-1, 3))

    for i in arange(index.shape[0]):
        steps[i]['args'][0]['visible'][index[i, 0]] = True
        steps[i]['args'][0]['visible'][index[i, 1]] = True
        steps[i]['args'][0]['visible'][index[i, 2]] = True

    sliders = [dict(
        active = 0,
        currentvalue = dict(prefix = 't = '),
        pad = dict(t = 50),
        steps = steps
    )]

    figure.update_layout(sliders = sliders, height = 1000,
        xaxis = dict(title = 'x [m]', range = [x['min'], x['max']], autorange = False),
        xaxis2 = dict(title = 'x [m]', range = [x['min'], x['max']], autorange = False), 
        xaxis3 = dict(title = 'x [m]', range = [x['min'], x['max']], autorange = False),
        yaxis = dict(title = 'y [m]', range = [y['min'], y['max']], autorange = False, scaleanchor = 'x', scaleratio = 1),
        yaxis2 = dict(title = 'y [m]', range = [y['min'], y['max']], autorange = False,scaleanchor = 'x2', scaleratio = 1), 
        yaxis3 = dict(title = 'y [m]', range = [y['min'], y['max']], autorange = False, scaleanchor = 'x3', scaleratio = 1)
    )

    return figure.to_json()

if __name__ == '__main__':

    sio = socketio.Client()
    sio.connect('http://localhost:4000')

    configurate = json.loads(sys.argv[1])

    # configurate = dict(x = [0, 250e-3, 10], y = [0, 50e-3, 10], t = [0, 0.001, 10], nu = [1], rho = [1], iter_res = [500], tol = [1e-2], iter_ref = [10])

    x = dict(node = configurate['x'][2], min = configurate['x'][0], max = configurate['x'][1])
    y = dict(node = configurate['y'][2], min = configurate['y'][0], max = configurate['y'][1])
    t = dict(node = configurate['t'][2], min = configurate['t'][0], max = configurate['t'][1])

    x['array'] = linspace(x['min'], x['max'], x['node'])
    y['array'] = linspace(y['min'], y['max'], y['node'])
    t['array'] = linspace(t['min'], t['max'], t['node'])

    x['matrix'], y['matrix'] = meshgrid(x['array'], y['array'])

    x['step'] = diff(x['array'])[0]
    y['step'] = diff(y['array'])[0]
    t['step'] = diff(t['array'])[0]

    dx = x['step']
    dy = y['step']
    dt = t['step']

    rho = configurate['rho'][0]
    nu = configurate['nu'][0]

    options = dict(iteration_restriction = configurate['iter_res'][0], tolerance = configurate['tol'][0], 
        iteration_reference = configurate['iter_ref'][0])

    nodes_domain = array([], dtype = int)
    nodes_inside = array([], dtype = int)
    nodes_boundary = array([], dtype = int)

    for i in arange(x['node']):
        for j in arange(y['node']):
            nodes_domain = append(nodes_domain, [i, j], axis = 0)
            if (i > 0 and i < x['node'] - 1) and (j > 0 and j < y['node'] - 1):
                nodes_inside = append(nodes_inside, [i, j], axis = 0)
            else:
                nodes_boundary = append(nodes_boundary, [i, j], axis = 0)

    nodes_domain = nodes_domain.reshape((x['node'] * y['node'], 2))
    nodes_inside = nodes_inside.reshape(((x['node'] - 2) * (y['node'] - 2), 2))

    nodes_boundary = stack((concatenate([arange(x['node']), ((x['node'] - 1) * ones(y['node'] - 2)), arange(x['node'] - 1, 0, -1), zeros(y['node'] - 1)]), 
        concatenate([zeros(x['node']), arange(1, y['node'] - 1), ((y['node'] - 1) * ones(x['node'] - 1)), arange(y['node'] - 1, 0, -1)]))).T

    nodes_boundary = nodes_boundary.astype(int)

    nodes = dict(domain = nodes_domain, inside = nodes_inside, boundary = nodes_boundary)

    nodes['domain'] = nodes_domain
    nodes['inside'] = nodes_inside
    nodes['boundary'] = nodes_boundary

    nodes_u = reshape(arange(x['node']*y['node']), [x['node'], y['node']])
    nodes_v = reshape(arange(x['node']*y['node'], 2 * x['node']*y['node']), [x['node'], y['node']])
    nodes_p = reshape(arange(2 * x['node']*y['node'], 3 * x['node']*y['node']), [x['node'], y['node']])

    u = zeros([x['node'], y['node'], t['node']])
    v = zeros([x['node'], y['node'], t['node']])
    p = zeros([x['node'], y['node'], t['node']])

    u[:, :, 1] = 0
    v[:, :, 1] = 0
    p[:, :, 1] = 0

    solutions = zeros([3 * x['node'] * y['node'], t['node']]);
    solutions[:, 0] = reshape([u[:, :, 0].flatten(), v[:, :, 0].flatten(), p[:, :, 0].flatten()], -1)
    argument_previously = solutions[:, 0]

    bc_u_k1 = zeros((x['node'], y['node']))
    bc_u_k21 = zeros((x['node'], y['node']))
    bc_u_k22 = zeros((x['node'], y['node']))
    bc_u_k3 = zeros((x['node'], y['node']))

    bc_v_k1 = zeros((x['node'], y['node']))
    bc_v_k21 = zeros((x['node'], y['node']))
    bc_v_k22 = zeros((x['node'], y['node']))
    bc_v_k3 = zeros((x['node'], y['node']))

    bc_p_k1 = zeros((x['node'], y['node']))
    bc_p_k21 = zeros((x['node'], y['node']))
    bc_p_k22 = zeros((x['node'], y['node']))
    bc_p_k3 = zeros((x['node'], y['node']))

    # inlet
    bc_u_k1[1:-1, 0] = 1
    bc_u_k3[1:-1, 0] = 1
    bc_v_k1[1:-1, 0] = 1
    bc_p_k22[1:-1, 0] = 1

    # outlet
    bc_u_k22[1:-1, -1] = -1
    bc_v_k22[1:-1, -1] = -1
    bc_p_k1[1:-1, -1] = -1

    # wall
    bc_u_k1[0, :] = 1
    bc_v_k1[0, :] = 1
    bc_p_k21[0, :] = 1

    # wall
    bc_u_k1[-1, :] = -1
    bc_v_k1[-1, :] = -1
    bc_p_k21[-1, :] = -1

    convergence = array([])

    dim = 2

    nb = numpy.copy(nodes_boundary)
    nb_ip1j = numpy.copy(nodes_boundary)
    nb_ijp1 = numpy.copy(nodes_boundary)

    for i in arange(nb.shape[0]):
        if (nb_ip1j[i, 0] == x['node'] - 1):
            nb_ip1j[i, 0] = nb_ip1j[i, 0] - 1
        else:
            nb_ip1j[i, 0] = nb_ip1j[i, 0] + 1
        if (nb_ijp1[i, 1] == y['node'] - 1):
            nb_ijp1[i, 1] = nb_ijp1[i, 1] - 1
        else:
            nb_ijp1[i, 1] = nb_ijp1[i, 1] + 1

    nb = [numpy.squeeze(val) for val in numpy.split(nb.T, dim, axis = 0)]
    nb_ip1j = [numpy.squeeze(val) for val in numpy.split(nb_ip1j.T, dim, axis = 0)]
    nb_ijp1 = [numpy.squeeze(val) for val in numpy.split(nb_ijp1.T, dim, axis = 0)]

    nb_u_ij = nodes_u[nb]
    nb_v_ij = nodes_v[nb]
    nb_p_ij = nodes_p[nb]

    nb_u_ip1j = nodes_u[nb_ip1j]
    nb_u_ijp1 = nodes_u[nb_ijp1]

    nb_v_ip1j = nodes_v[nb_ip1j]
    nb_v_ijp1 = nodes_v[nb_ijp1]

    nb_p_ip1j = nodes_p[nb_ip1j]
    nb_p_ijp1 = nodes_p[nb_ijp1]

    ni = numpy.copy(nodes_inside)
    ni_ip1j = numpy.copy(nodes_inside)
    ni_im1j = numpy.copy(nodes_inside)
    ni_ijp1 = numpy.copy(nodes_inside)
    ni_ijm1 = numpy.copy(nodes_inside)

    ni_ip1j[:, 0] = ni_ip1j[:, 0] + 1
    ni_im1j[:, 0] = ni_im1j[:, 0] - 1
    ni_ijp1[:, 1] = ni_ijp1[:, 1] + 1
    ni_ijm1[:, 1] = ni_ijm1[:, 1] - 1

    ni = [numpy.squeeze(val) for val in numpy.split(ni.T, dim, axis = 0)]
    ni_ip1j = [numpy.squeeze(val) for val in numpy.split(ni_ip1j.T, dim, axis = 0)]
    ni_im1j = [numpy.squeeze(val) for val in numpy.split(ni_im1j.T, dim, axis = 0)]
    ni_ijp1 = [numpy.squeeze(val) for val in numpy.split(ni_ijp1.T, dim, axis = 0)]
    ni_ijm1 = [numpy.squeeze(val) for val in numpy.split(ni_ijm1.T, dim, axis = 0)]

    ni_u_ij = nodes_u[ni]
    ni_v_ij = nodes_v[ni]
    ni_p_ij = nodes_p[ni]

    ni_u_ip1j = nodes_u[ni_ip1j]
    ni_u_im1j = nodes_u[ni_im1j]
    ni_u_ijp1 = nodes_u[ni_ijp1]
    ni_u_ijm1 = nodes_u[ni_ijm1]

    ni_v_ip1j = nodes_v[ni_ip1j]
    ni_v_im1j = nodes_v[ni_im1j]
    ni_v_ijp1 = nodes_v[ni_ijp1]
    ni_v_ijm1 = nodes_v[ni_ijm1]

    ni_p_ip1j = nodes_p[ni_ip1j]
    ni_p_im1j = nodes_p[ni_im1j]
    ni_p_ijp1 = nodes_p[ni_ijp1]
    ni_p_ijm1 = nodes_p[ni_ijm1]

    # @jit
    def global_system (arg, arg_p):

        # arg_p = arg_p[0]

        result = zeros(argument_previously.size)

        u_ij = arg[nb_u_ij]
        v_ij = arg[nb_v_ij]
        p_ij = arg[nb_p_ij]

        u_ip1j = arg[nb_u_ip1j]
        u_ijp1 = arg[nb_u_ijp1]

        v_ip1j = arg[nb_v_ip1j]
        v_ijp1 = arg[nb_v_ijp1]

        p_ip1j = arg[nb_p_ip1j]
        p_ijp1 = arg[nb_p_ijp1]

        result[nb_u_ij] = bc_u_k1[nb] * u_ij + bc_u_k21[nb] * (u_ip1j - u_ij) / dx + bc_u_k22[nb] * (u_ijp1 - u_ij) / dy - bc_u_k3[nb]
        result[nb_v_ij] = bc_v_k1[nb] * v_ij + bc_v_k21[nb] * (v_ip1j - v_ij) / dx + bc_v_k22[nb] * (v_ijp1 - v_ij) / dy - bc_v_k3[nb]
        result[nb_p_ij] = bc_p_k1[nb] * p_ij + bc_p_k21[nb] * (p_ip1j - p_ij) / dx + bc_p_k22[nb] * (p_ijp1 - p_ij) / dy - bc_p_k3[nb]

        u_ij = arg[ni_u_ij]
        v_ij = arg[ni_v_ij]
        p_ij = arg[ni_p_ij]

        u_ij_p = arg_p[ni_u_ij]
        v_ij_p = arg_p[ni_v_ij]

        u_ip1j = arg[ni_u_ip1j]
        u_im1j = arg[ni_u_im1j]
        u_ijp1 = arg[ni_u_ijp1]
        u_ijm1 = arg[ni_u_ijm1]

        v_ip1j = arg[ni_v_ip1j]
        v_im1j = arg[ni_v_im1j]
        v_ijp1 = arg[ni_v_ijp1]
        v_ijm1 = arg[ni_v_ijm1]

        p_ip1j = arg[ni_p_ip1j]
        p_im1j = arg[ni_p_im1j]
        p_ijp1 = arg[ni_p_ijp1]
        p_ijm1 = arg[ni_p_ijm1]

        result[ni_u_ij] = ((u_ij - u_ij_p) / dt + u_ij * (u_ip1j - u_im1j) / (2 * dx) + 
            v_ij * (u_ijp1 - u_ijm1) / (2 * dy) + 1 / rho * (p_ip1j - p_im1j) / (2 * dx) - 
            nu * ((u_ip1j - 2 * u_ij + u_im1j) / (dx)**2 + (u_ijp1 - 2 * u_ij + u_ijm1) / (dy)**2))
        result[ni_v_ij] = ((v_ij - v_ij_p) / dt + u_ij * (v_ip1j - v_im1j) / (2 * dx) + 
            v_ij * (v_ijp1 - v_ijm1) / (2 * dy) + 1 / rho * (p_ijp1 - p_ijm1) / (2 * dy) - 
            nu * ((v_ip1j - 2 * v_ij + v_im1j) / (dx)**2 + (v_ijp1 - 2 * v_ij + v_ijm1) / (dy)**2))
        result[ni_p_ij] = ((p_ip1j - 2 * p_ij + p_im1j) / (dx)**2 + (p_ijp1 - 2 * p_ij + p_ijm1) / (dy)**2 + 
            rho * (((u_ip1j - u_im1j) / (2 * dx))**2 + ((v_ijp1 - v_ijm1) / (2 * dy))**2 + 
            2 * (u_ijp1 - u_ijm1) / (2 * dy) * (v_ip1j - v_im1j) / (2 * dx)))

        return result

    # @jit
    def jacobian_system(arg):

        result = zeros([arg.size, arg.size])

        result[nb_u_ij, nb_u_ij] = bc_u_k1[nb] - bc_u_k21[nb] / dx - bc_u_k22[nb] / dy
        result[nb_u_ij, nb_u_ip1j] = bc_u_k21[nb] / dx
        result[nb_u_ij, nb_u_ijp1] = bc_u_k22[nb] / dy

        result[nb_v_ij, nb_v_ij] = bc_v_k1[nb] - bc_v_k21[nb] / dx - bc_v_k22[nb] / dy
        result[nb_v_ij, nb_v_ip1j] = bc_v_k21[nb] / dx
        result[nb_v_ij, nb_v_ijp1] = bc_v_k22[nb] / dy

        result[nb_p_ij, nb_p_ij] = bc_p_k1[nb] - bc_p_k21[nb] / dx - bc_p_k22[nb] / dy
        result[nb_p_ij, nb_p_ip1j] = bc_p_k21[nb] / dx
        result[nb_p_ij, nb_p_ijp1] = bc_p_k22[nb] / dy

        u_ij = arg[ni_u_ij]
        v_ij = arg[ni_v_ij]

        u_ip1j = arg[ni_u_ip1j]
        u_im1j = arg[ni_u_im1j]
        u_ijp1 = arg[ni_u_ijp1]
        u_ijm1 = arg[ni_u_ijm1]

        v_ip1j = arg[ni_v_ip1j]
        v_im1j = arg[ni_v_im1j]
        v_ijp1 = arg[ni_v_ijp1]
        v_ijm1 = arg[ni_v_ijm1]

        result[ni_u_ij, ni_u_ij] = 1 / dt + (u_ip1j - u_im1j) / (2 * dx) + nu * (2 / (dx)**2 + 2 / (dy)**2)
        result[ni_u_ij, ni_u_ip1j] = u_ij / (2 * dx) - nu * (1 / (dx)**2)
        result[ni_u_ij, ni_u_im1j] = (-1) * u_ij / (2 * dx) - nu * (1 / (dx)**2)
        result[ni_u_ij, ni_u_ijp1] = v_ij / (2 * dy) - nu * (1 / (dy)**2)
        result[ni_u_ij, ni_u_ijm1] = (-1) * v_ij / (2 * dy) - nu * (1 / (dy)**2)
        result[ni_u_ij, ni_v_ij] = (u_ijp1 - u_ijm1) / (2 * dy)
        result[ni_u_ij, ni_p_ip1j] = 1 / rho / (2 * dx)
        result[ni_u_ij, ni_p_im1j] = (-1) / rho / (2 * dx)

        result[ni_v_ij, ni_v_ij] = 1 / dt + (v_ijp1 - v_ijm1) / (2 * dy) + nu * (2 / (dx)**2 + 2 / (dy)**2)
        result[ni_v_ij, ni_v_ip1j] = u_ij / (2 * dx) - nu * (1 / (dx)**2)
        result[ni_v_ij, ni_v_im1j] = (-1) * u_ij / (2 * dx) - nu * (1 / (dx)**2)
        result[ni_v_ij, ni_v_ijp1] = v_ij / (2 * dy) - nu * (1 / (dy)**2)
        result[ni_v_ij, ni_v_im1j] = (-1) * v_ij / (2 * dy) - nu * (1 / (dy)**2)     
        result[ni_v_ij, ni_u_ij] = (v_ip1j - v_im1j) / (2 * dx)
        result[ni_v_ij, ni_p_ijp1] = 1 / rho / (2 * dy)
        result[ni_v_ij, ni_p_ijm1] = (-1) / rho / (2 * dy)

        result[ni_p_ij, ni_p_ij] = (-2) / (dx)**2 + (-2) / (dy)**2
        result[ni_p_ij, ni_p_ip1j] = 1 / (dx)**2
        result[ni_p_ij, ni_p_im1j] = 1 / (dx)**2
        result[ni_p_ij, ni_p_ijp1] = 1 / (dy)**2
        result[ni_p_ij, ni_p_ijm1] = 1 / (dy)**2
        result[ni_p_ij, ni_u_ip1j] = rho * 2 / (2 * dx) * (u_ip1j - u_im1j)
        result[ni_p_ij, ni_u_im1j] = (-1) * rho * 2 / (2 * dx) * (u_ip1j - u_im1j)
        result[ni_p_ij, ni_v_ijp1] = rho * 2 / (2 * dy) * (v_ijp1 - v_ijm1)
        result[ni_p_ij, ni_v_ijm1] = (-1) * rho * 2 / (2 * dy) * (v_ijp1 - v_ijm1)
        result[ni_p_ij, ni_u_ijp1] = 2 * (v_ip1j - v_im1j) / (2 * dx) / (2 * dy)
        result[ni_p_ij, ni_u_ijm1] = (-2) * (v_ip1j - v_im1j) / (2 * dx) / (2 * dy)
        result[ni_p_ij, ni_v_ip1j] = 2 * (u_ijp1 - u_ijm1) / (2 * dy) / (2 * dx)
        result[ni_p_ij, ni_v_im1j] = (-2) * (u_ijp1 - u_ijm1) / (2 * dy) / (2 * dx)

        return result

    # @jit
    def newton (initial):

        tolerance, iteration_convergence, iteration_calculation = 0, 0, 0
        convergence, calculation = False, True
        argument_previously = initial

        while calculation:

            delta = linsolve(jacobian_system(initial), (-1) * global_system(initial, argument_previously))
            solution = delta + initial
            tolerance = sqrt(sum((abs(solution - initial))**2))

            if (iteration_calculation <= options['iteration_restriction']):
                if (tolerance < options['tolerance']):
                    iteration_convergence = iteration_convergence + 1
                else:
                    iteration_convergence = 0
                if (iteration_convergence >= options['iteration_reference']):
                    convergence, calculation = True, False
            else:
                convergence, calculation = False, False

            sio.emit('figure-process', json.dumps(dict(tolerance = tolerance)))

            initial = solution
            iteration_calculation = iteration_calculation + 1

        return solution, convergence

    for i in arange(1, t['node']):
        solutions[:, i], convergence = newton(solutions[:, i - 1])
        # solutions[:, i] = fsolve(global_system, solutions[:, i - 1], args = (solutions[:, i - 1],))

    for k in arange(1, t['node']):
        for i, j in nodes['domain']:
            u[i, j, k] = solutions[nodes_u[i, j], k]
            v[i, j, k] = solutions[nodes_v[i, j], k]
            p[i, j, k] = solutions[nodes_p[i, j], k]

    sio.emit('figure-result', fig2json(x, y, t, u, v, p))