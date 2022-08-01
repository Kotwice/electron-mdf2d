
from numpy import linspace, zeros, meshgrid, diff, ones, arange, array, append, reshape, concatenate, stack, sqrt, split, array_split, sum, abs
from numpy.matlib import repmat
from matplotlib import pyplot as plt
from scipy.linalg import solve as linsolve
from scipy.optimize import fsolve
import multiprocessing as mp
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numba import jit
import json, socketio

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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@jit
def global_system (argument, parameters):

    x = parameters['x']
    y = parameters['y']
    t = parameters['t']

    rho = parameters['rho']
    nu = parameters['nu']

    nodes_inside = parameters['nodes_inside']
    nodes_boundary = parameters['nodes_boundary']
    nodes_u = parameters['nodes_u']
    nodes_v = parameters['nodes_v']
    nodes_p = parameters['nodes_p']
    argument_previously = parameters['argument_previously']
    
    bc_u_k1 = parameters['bc_u_k1']
    bc_u_k21 = parameters['bc_u_k21']
    bc_u_k22 = parameters['bc_u_k22']
    bc_u_k3 = parameters['bc_u_k3']

    bc_v_k1 = parameters['bc_v_k1']
    bc_v_k21 = parameters['bc_v_k21']
    bc_v_k22 = parameters['bc_v_k22']
    bc_v_k3 = parameters['bc_v_k3']

    bc_p_k1 = parameters['bc_p_k1']
    bc_p_k21 = parameters['bc_p_k21']
    bc_p_k22 = parameters['bc_p_k22']
    bc_p_k3 = parameters['bc_p_k3']

    result = zeros(argument_previously.size)

    for index in arange(nodes_boundary.shape[0]):

        i = nodes_boundary[index, 0]
        j = nodes_boundary[index, 1]

        u_ij = argument[nodes_u[i, j]]
        v_ij = argument[nodes_v[i, j]]
        p_ij = argument[nodes_p[i, j]]

        if (i == x['node'] - 1):
            u_i1j = argument[nodes_u[i, j]]
            u_i0j = argument[nodes_u[i - 1, j]]
            v_i1j = argument[nodes_v[i, j]]
            v_i0j = argument[nodes_v[i - 1, j]]
            p_i1j = argument[nodes_p[i, j]]
            p_i0j = argument[nodes_p[i - 1, j]]
        else:
            u_i1j = argument[nodes_u[i + 1, j]]
            u_i0j = argument[nodes_u[i, j]]
            v_i1j = argument[nodes_v[i + 1, j]]
            v_i0j = argument[nodes_v[i, j]]
            p_i1j = argument[nodes_p[i + 1, j]]
            p_i0j = argument[nodes_p[i, j]]

        if (j == y['node'] - 1):
            u_ij1 = argument[nodes_u[i, j]]
            u_ij0 = argument[nodes_u[i, j - 1]]
            v_ij1 = argument[nodes_v[i, j]]
            v_ij0 = argument[nodes_v[i, j - 1]]
            p_ij1 = argument[nodes_p[i, j]]
            p_ij0 = argument[nodes_p[i, j - 1]]
        else:
            u_ij1 = argument[nodes_u[i, j + 1]]
            u_ij0 = argument[nodes_u[i, j]]
            v_ij1 = argument[nodes_v[i, j + 1]]
            v_ij0 = argument[nodes_v[i, j]]
            p_ij1 = argument[nodes_p[i, j + 1]]
            p_ij0 = argument[nodes_p[i, j]]

        result[nodes_u[i, j]] = bc_u_k1[index] * u_ij + bc_u_k21[index] * (u_i1j - u_i0j) / x['step'] + bc_u_k22[index] * (u_ij1 - u_ij0) / y['step'] - bc_u_k3[index]
        result[nodes_v[i, j]] = bc_v_k1[index] * v_ij + bc_v_k21[index] * (v_i1j - v_i0j) / x['step'] + bc_v_k22[index] * (v_ij1 - v_ij0) / y['step'] - bc_v_k3[index]
        result[nodes_p[i, j]] = bc_p_k1[index] * p_ij + bc_p_k21[index] * (p_i1j - p_i0j) / x['step'] + bc_p_k22[index] * (p_ij1 - p_ij0) / y['step'] - bc_p_k3[index]


    for i, j in nodes_inside:

        u_ij = argument[nodes_u[i, j]]
        v_ij = argument[nodes_v[i, j]]
        p_ij = argument[nodes_p[i, j]]

        u_ij_p = argument_previously[nodes_u[i, j]]
        v_ij_p = argument_previously[nodes_v[i, j]]

        u_i1pj = argument[nodes_u[i + 1, j]]
        u_i1mj = argument[nodes_u[i - 1, j]]
        u_ij1p = argument[nodes_u[i, j + 1]]
        u_ij1m = argument[nodes_u[i, j - 1]]

        v_i1pj = argument[nodes_v[i + 1, j]]
        v_i1mj = argument[nodes_v[i - 1, j]]
        v_ij1p = argument[nodes_v[i, j + 1]]
        v_ij1m = argument[nodes_v[i, j - 1]]

        p_i1pj = argument[nodes_p[i + 1, j]]
        p_i1mj = argument[nodes_p[i - 1, j]]
        p_ij1p = argument[nodes_p[i, j + 1]]
        p_ij1m = argument[nodes_p[i, j - 1]]

        result[nodes_u[i, j]] = ((u_ij - u_ij_p) / t['step'] + u_ij * (u_i1pj - u_i1mj) / (2 * x['step']) + 
            v_ij * (u_ij1p - u_ij1m) / (2 * y['step']) + 1 / rho * (p_i1pj - p_i1mj) / (2 * x['step']) - 
            nu * ((u_i1pj - 2 * u_ij + u_i1mj) / (x['step'])**2 + (u_ij1p - 2 * u_ij + u_ij1m) / (y['step'])**2))
        result[nodes_v[i, j]] = ((v_ij - v_ij_p) / t['step'] + u_ij * (v_i1pj - v_i1mj) / (2 * x['step']) + 
            v_ij * (v_ij1p - v_ij1m) / (2 * y['step']) + 1 / rho * (p_ij1p - p_ij1m) / (2 * y['step']) - 
            nu * ((v_i1pj - 2 * v_ij + v_i1mj) / (x['step'])**2 + (v_ij1p - 2 * v_ij + v_ij1m) / (y['step'])**2))
        result[nodes_p[i, j]] = ((p_i1pj - 2 * p_ij + p_i1mj) / (x['step'])**2 + (p_ij1p - 2 * p_ij + p_ij1m) / (y['step'])**2 + 
            rho * (((u_i1pj - u_i1mj) / (2 * x['step']))**2 + ((v_ij1p - v_ij1m) / (y['step']))**2 + 
            2 * (u_ij1p - u_ij1m) / (y['step']) * (v_i1pj - v_i1mj) / (x['step'])))

    return result

@jit
def jacobian_system(argument, parameters):

    x = parameters['x']
    y = parameters['y']
    t = parameters['t']

    rho = parameters['rho']
    nu = parameters['nu']

    nodes_inside = parameters['nodes_inside']
    nodes_boundary = parameters['nodes_boundary']
    nodes_u = parameters['nodes_u']
    nodes_v = parameters['nodes_v']
    nodes_p = parameters['nodes_p']
    
    bc_u_k1 = parameters['bc_u_k1']
    bc_u_k21 = parameters['bc_u_k21']
    bc_u_k22 = parameters['bc_u_k22']

    bc_v_k1 = parameters['bc_v_k1']
    bc_v_k21 = parameters['bc_v_k21']
    bc_v_k22 = parameters['bc_v_k22']

    bc_p_k1 = parameters['bc_p_k1']
    bc_p_k21 = parameters['bc_p_k21']
    bc_p_k22 = parameters['bc_p_k22']

    result = zeros([argument.size, argument.size])

    for index in arange(nodes_boundary.shape[0]):

        i = nodes_boundary[index, 0]
        j = nodes_boundary[index, 1]

        if (i == x['node'] - 1):
            k_x = 1
        else:
            k_x = - 1
        if (j == y['node'] - 1):
            k_y = 1
        else:
            k_y = - 1

        result[nodes_u[i, j], nodes_u[i, j]] = bc_u_k1[index] + k_x * bc_u_k21[index] / x['step'] + k_y * bc_u_k22[index] / y['step']
        result[nodes_u[i, j], nodes_u[i - k_x, j]] = - k_x * bc_u_k21[index] / x['step']
        result[nodes_u[i, j], nodes_u[i, j - k_y]] = - k_y * bc_u_k22[index] / y['step']

        result[nodes_v[i, j], nodes_v[i, j]] = bc_v_k1[index] + k_x * bc_v_k21[index] / x['step'] + k_y * bc_v_k22[index] / y['step']
        result[nodes_v[i, j], nodes_v[i - k_x, j]] = - k_x * bc_v_k21[index] / x['step']
        result[nodes_v[i, j], nodes_v[i, j - k_y]] = - k_y * bc_v_k22[index] / y['step']

        result[nodes_p[i, j], nodes_p[i, j]] = bc_p_k1[index] + k_x * bc_p_k21[index] / x['step'] + k_y * bc_p_k22[index] / y['step']
        result[nodes_p[i, j], nodes_p[i - k_x, j]] = - k_x * bc_p_k21[index] / x['step']
        result[nodes_p[i, j], nodes_p[i, j - k_y]] = - k_y * bc_p_k22[index] / y['step']

    for i, j in nodes_inside:

        u_ij = argument[nodes_u[i, j]]
        v_ij = argument[nodes_v[i, j]]

        u_i1pj = argument[nodes_u[i + 1, j]]
        u_i1mj = argument[nodes_u[i - 1, j]]
        u_ij1p = argument[nodes_u[i, j + 1]]
        u_ij1m = argument[nodes_u[i, j - 1]]

        v_i1pj = argument[nodes_v[i + 1, j]]
        v_i1mj = argument[nodes_v[i - 1, j]]
        v_ij1p = argument[nodes_v[i, j + 1]]
        v_ij1m = argument[nodes_v[i, j - 1]]

        result[nodes_u[i, j], nodes_u[i, j]] = 1 / t['step'] + (u_i1pj - u_i1mj) / (2 * x['step']) + nu * (2 / (x['step'])**2 + 2 / (y['step'])**2)
        result[nodes_u[i, j], nodes_u[i + 1, j]] = u_ij / (2 * x['step']) - nu * (1 / (x['step'])**2)
        result[nodes_u[i, j], nodes_u[i - 1, j]] = (-1) * u_ij / (2 * x['step']) - nu * (1 / (x['step'])**2)
        result[nodes_u[i, j], nodes_u[i, j + 1]] = v_ij / (2 * y['step']) - nu * (1 / (y['step'])**2)
        result[nodes_u[i, j], nodes_u[i - 1, j]] = (-1) * v_ij / (2 * y['step']) - nu * (1 / (y['step'])**2)
        result[nodes_u[i, j], nodes_v[i, j]] = (u_ij1p - u_ij1m) / (2 * y['step'])
        result[nodes_u[i, j], nodes_p[i + 1, j]] = 1 / rho / (2 * x['step'])
        result[nodes_u[i, j], nodes_p[i - 1, j]] = (-1) / rho / (2 * x['step'])

        result[nodes_v[i, j], nodes_v[i, j]] = 1 / t['step'] + (v_ij1p - v_ij1m) / (2 * y['step']) + nu * (2 / (x['step'])**2 + 2 / (y['step'])**2)
        result[nodes_v[i, j], nodes_v[i + 1, j]] = u_ij / (2 * x['step']) - nu * (1 / (x['step'])**2)
        result[nodes_u[i, j], nodes_u[i - 1, j]] = (-1) * u_ij / (2 * x['step']) - nu * (1 / (x['step'])**2)
        result[nodes_u[i, j], nodes_u[i, j + 1]] = v_ij / (2 * y['step']) - nu * (1 / (y['step'])**2)
        result[nodes_u[i, j], nodes_u[i - 1, j]] = (-1) * v_ij / (2 * y['step']) - nu * (1 / (y['step'])**2)
        result[nodes_v[i, j], nodes_u[i, j]] = (v_i1pj - v_i1mj) / (2 * x['step'])
        result[nodes_v[i, j], nodes_p[i, j + 1]] = 1 / rho / (2 * y['step'])
        result[nodes_v[i, j], nodes_p[i, j - 1]] = (-1) / rho / (2 * y['step'])

        result[nodes_p[i, j], nodes_p[i, j]] = (-2) / (x['step'])**2 + (-2) / (y['step'])**2
        result[nodes_p[i, j], nodes_p[i + 1, j]] = 1 / (x['step'])**2
        result[nodes_p[i, j], nodes_p[i - 1, j]] = 1 / (x['step'])**2
        result[nodes_p[i, j], nodes_p[i, j + 1]] = 1 / (y['step'])**2
        result[nodes_p[i, j], nodes_p[i, j - 1]] = 1 / (y['step'])**2
        result[nodes_p[i, j], nodes_u[i + 1, j]] = rho * 2 / (2 * x['step'])**2 * (u_i1pj - u_i1mj)
        result[nodes_p[i, j], nodes_u[i - 1, j]] = (-1) * rho * 2 / (2 * x['step'])**2 * (u_i1pj - u_i1mj)
        result[nodes_p[i, j], nodes_u[i, j + 1]] = 2 / (y['step']) * (v_i1pj - v_i1mj) / (x['step'])
        result[nodes_p[i, j], nodes_u[i, j - 1]] = (-2) / (y['step']) * (v_i1pj - v_i1mj) / (x['step'])     
        result[nodes_p[i, j], nodes_v[i + 1, j]] = rho * 2 / (2 * y['step'])**2 * (v_ij1p - v_ij1m)
        result[nodes_p[i, j], nodes_v[i - 1, j]] = (-1) * rho * 2 / (2 * y['step'])**2 * (v_ij1p - v_ij1m)
        result[nodes_p[i, j], nodes_v[i, j + 1]] = 2 * (u_ij1p - u_ij1m) / (y['step']) / (x['step'])
        result[nodes_p[i, j], nodes_v[i, j - 1]] = (-2) * (u_ij1p - u_ij1m) / (y['step']) / (x['step'])

    return result