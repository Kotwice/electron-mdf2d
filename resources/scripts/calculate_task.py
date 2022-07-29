from numpy import linspace, zeros, meshgrid, diff, ones, arange, array, append, reshape, concatenate, stack, sqrt, split, array_split, sum, abs
from scipy.linalg import solve as linsolve

import sys, json, socketio, kotlib, sys

if __name__ == '__main__':

    sio = socketio.Client()
    sio.connect('http://localhost:4000')

    configurate = json.loads(sys.argv[1])

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

    rho = configurate['rho'][0]
    nu = configurate['nu'][0]

    options = dict(iteration_restriction = configurate['iter_res'][0], tolerance = configurate['tol'][0], 
        iteration_reference = configurate['iter_ref'][0])

    u_0 = 0 * ones([x['node'], y['node']])
    v_0 = 0 * ones([x['node'], y['node']])
    p_0 = 0 * ones([x['node'], y['node']])

    bc_u_k1 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_u_k21 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_u_k22 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_u_k3 = zeros(2 * (x['node'] + y['node']) - 4)

    bc_v_k1 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_v_k21 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_v_k22 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_v_k3 = zeros(2 * (x['node'] + y['node']) - 4)

    bc_p_k1 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_p_k21 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_p_k22 = zeros(2 * (x['node'] + y['node']) - 4)
    bc_p_k3 = zeros(2 * (x['node'] + y['node']) - 4)

    indexes_west = arange(x['node'], dtype = int)
    indexes_east = arange(x['node'] + y['node'] - 2, 2 * x['node'] + y['node'] - 2, dtype = int)
    indexes_south = arange(2 * x['node'] + y['node'] - 2, 2 * (x['node'] + y['node']) - 4, dtype = int)
    indexes_north = arange(x['node'], x['node'] + y['node'] - 2, dtype = int)

    # inlet

    indexes_west_split = array_split(indexes_west, 2)

    bc_u_k1[indexes_west] = 1
    bc_u_k21[indexes_west] = 0
    bc_u_k22[indexes_west] = 0
    # bc_u_k3[indexes_west] = 1

    bc_u_k3[indexes_west_split[0]] = 0
    bc_u_k3[indexes_west_split[1]] = 1

    bc_v_k1[indexes_west] = 1
    bc_v_k21[indexes_west] = 0
    bc_v_k22[indexes_west] = 0
    bc_v_k3[indexes_west] = 0

    bc_p_k1[indexes_west] = 0
    bc_p_k21[indexes_west] = 0
    bc_p_k22[indexes_west] = 1
    bc_p_k3[indexes_west] = 0

    # outlet
    bc_u_k1[indexes_east] = 0
    bc_u_k21[indexes_east] = 0
    bc_u_k22[indexes_east] = 1
    bc_u_k3[indexes_east] = 0

    bc_v_k1[indexes_east] = 0
    bc_v_k21[indexes_east] = 0
    bc_v_k22[indexes_east] = 1
    bc_v_k3[indexes_east] = 0

    bc_p_k1[indexes_east] = 1
    bc_p_k21[indexes_east] = 0
    bc_p_k22[indexes_east] = 0
    bc_p_k3[indexes_east] = 0

    # bottom wall
    bc_u_k1[indexes_south] = 1
    bc_u_k21[indexes_south] = 0
    bc_u_k22[indexes_south] = 0
    bc_u_k3[indexes_south] = 0

    bc_v_k1[indexes_south] = 1
    bc_v_k21[indexes_south] = 0
    bc_v_k22[indexes_south] = 0
    bc_v_k3[indexes_south] = 0

    bc_p_k1[indexes_south] = 0
    bc_p_k21[indexes_south] = 1
    bc_p_k22[indexes_south] = 0
    bc_p_k3[indexes_south] = 0

    # upper wall
    bc_u_k1[indexes_north] = 0
    bc_u_k21[indexes_north] = 1
    bc_u_k22[indexes_north] = 0
    bc_u_k3[indexes_north] = 0

    bc_v_k1[indexes_north] = 0  
    bc_v_k21[indexes_north] = 1
    bc_v_k22[indexes_north] = 0
    bc_v_k3[indexes_north] = 0  

    bc_p_k1[indexes_north] = 0
    bc_p_k21[indexes_north] = 1 
    bc_p_k22[indexes_north] = 0
    bc_p_k3[indexes_north] = 0

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

    u[:, :, 1] = u_0
    v[:, :, 1] = v_0
    p[:, :, 1] = p_0

    solutions = zeros([3 * x['node'] * y['node'], t['node']]);
    solutions[:, 0] = reshape([u[:, :, 0].flatten(), v[:, :, 0].flatten(), p[:, :, 0].flatten()], -1)

    convergence = array([])
    parameters = dict()
    parameters['x'] = x
    parameters['y'] = y
    parameters['t'] = t
    parameters['rho'] = rho
    parameters['nu'] = nu
    parameters['nodes'] = nodes
    parameters['nodes_inside'] = nodes_inside
    parameters['nodes_boundary'] = nodes_boundary
    parameters['nodes_u'] = nodes_u
    parameters['nodes_v'] = nodes_v
    parameters['nodes_p'] = nodes_p
    parameters['bc_u_k1'] = bc_u_k1
    parameters['bc_u_k21'] = bc_u_k21
    parameters['bc_u_k22'] = bc_u_k22
    parameters['bc_u_k3'] = bc_u_k3
    parameters['bc_v_k1'] = bc_v_k1
    parameters['bc_v_k21'] = bc_v_k21
    parameters['bc_v_k22'] = bc_v_k22
    parameters['bc_v_k3'] = bc_v_k3
    parameters['bc_p_k1'] = bc_p_k1
    parameters['bc_p_k21'] = bc_p_k21
    parameters['bc_p_k22'] = bc_p_k22
    parameters['bc_p_k3'] = bc_p_k3

    def newton (func, initial, jacob, options):

        tolerance, iteration_convergence, iteration_calculation = 0, 0, 0
        convergence, calculation = False, True

        while calculation:

            delta = linsolve(jacob(initial), (-1) * func(initial))
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
        parameters['argument_previously'] = solutions[:, i - 1]
        fun = lambda argument : kotlib.global_system (argument, parameters)
        jacob = lambda argument : kotlib.jacobian_system (argument, parameters)
        solutions[:, i], convergence = newton(fun, solutions[:, i - 1], jacob, options)

    for k in arange(1, t['node']):
        for i, j in nodes['domain']:
            u[i, j, k] = solutions[nodes_u[i, j], k]
            v[i, j, k] = solutions[nodes_v[i, j], k]
            p[i, j, k] = solutions[nodes_p[i, j], k]

    sio.emit('figure-result', kotlib.fig2json(x, y, t, u, v, p))