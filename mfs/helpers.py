import numpy as np
from multiprocessing import Pool as ThreadPool 
import multiprocessing


ncpu = multiprocessing.cpu_count()



'''Rotation functions'''
def rotate_around_z(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],[-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
def rotate_around_x(phi):
    return np.array([[1,0,0],[0,np.cos(phi), -np.sin(phi)],[0,np.sin(phi),np.cos(phi)]])
def rotate_to_dashed_frame(r, theta, phi):
    return np.dot(np.dot(rotate_around_z(theta), rotate_around_x(phi)), r)
def rotate_to_normal_frame(rDash, theta, phi):
    return np.dot(np.dot(rotate_around_x(-phi), rotate_around_z(-theta)), rDash)




'''Projection functions'''
def evaluate_axis_projection(projection):
    '''Interpret a 2D projection request
    
    Interprets a string containing the unique letters 'xyz' into indicies for
    reading values out of an array and plotting a projection. The first letter \
    will go on the graph's X-axis, the 2nd letter on the graph's Y axis.
    
    For example, 'zyx' or 'zy' will result in projecting the simulation's Z-axis
    onto the horizontal axis of a 2D graph (and label it accordingly), and the
    simulation's Y-axis onto the vertical axis of the 2D graph. 
    
    Parameters
    ----------
    projection : str
        Desired projection order
        
    Returns
    -------
    axOnePos : int
        The index of the simulation axis that will be placed on the graph's 
        first (x) axis. For example, if the string was 'zyx', this will have
        the value 2
    axTwoPos : int
        The index of the simulation axis that will be placed on the graph's 
        second (y) axis. For example, if the string was 'zyx', this will have
        the value 1
    axThreePos : int
        The index of the simulation axis that will be placed on the graph's 
        third (z) axis. For example, if the string was 'zyx', this will have
        the value 0
    '''
    proj = list(projection.lower())
    try:
        if proj[0] == 'x':
            axOnePos = 0
        elif proj[0] == 'y':
            axOnePos = 1
        elif proj[0] == 'z':
            axOnePos = 2
        if len(proj) > 1:
            if proj[1] == 'x':
                axTwoPos = 0
            elif proj[1] == 'y':
                axTwoPos = 1
            elif proj[1] == 'z':
                axTwoPos = 2
            if len(proj) > 2:
                if proj[2] == 'x':
                    axThreePos = 0
                elif proj[2] == 'y':
                    axThreePos = 1
                elif proj[2] == 'z':
                    axThreePos = 2
            else:
                axThreePos = 3-(axOnePos + axTwoPos)
        else:
            axTwoPos = 3 - axOnePos
            axThreePos = 3-(axOnePos + axTwoPos)
        return axOnePos, axTwoPos, axThreePos
    except Exception as e:
        print('Your projection string was not understood. Please provide a 1, 2, or 3 length string containing the letters x, y, z')
        print(e)









def plot_scalar_B_field(magnets, axes, centre, limit, projection, points):
    a1p, a2p, a3p = evaluate_axis_projection(projection) # This converts the projection into indicies. Here, we only care about the first index, the 2nd and 3rd are just dummies
    axOneLimLow = centre[a1p] - limit
    axOneLimHigh = centre[a1p] + limit
    axis = np.linspace(axOneLimLow, axOneLimHigh, points)
    B_field = np.zeros(points)
    for a, pos in enumerate(axis):
        r0 = centre[:]
        r0[a1p] = pos
        B_field_0 = np.zeros(3)
        for m in magnets:
            B_field_0 += m.get_B_field(r0)
        B_field[a] += np.linalg.norm(B_field_0)
    axes.plot(axis, B_field)
    axes.set_xlabel("%s [m]" % 'xyz'[a1p])
    axes.set_ylabel("B field [T]")
    return axis, B_field






def plot_vector_B_field(magnets, axes, centre, limit, projection, points=50, threads=ncpu):
    '''Produce a 50x50 grid of vector arrows for the magnetic field arising from [magnets]
    It produces a projection of the field based on the 'projection' parameter around the point (centre)
    This function divides the positions and magnets up among multiple threads. It's a dramatic slow down for single magnet (roughly /12 slower) due to 
    the overhead, but a fairly substantial speedup (x3) for a large number.
    
    Parameters
    ----------
    magnets : mfs.Magnet or list of mfs.Magnet
        All magnet sources to include in the calculation. Should be descendents of mfs.Magnet
    axes : matplotlib.Axes
        Matplotlib axes objects on which to display the resulting figure
    centre : np.ndarray
        Origin of the grid over which the B field is calculated: r=(x, y, z)
    limit : float
        half-width of the grid over which B-field is calculated
        The total grid is calculated as (r-limit, r+limit)
    projection : str
        Projection over which to evaluate the B-field, see mfs.helpers.evaluate_axis_projection
    points : int
        Number of points along each edge of the grid. Computational complexity scales as O(points^2)
    threads : int
        Number of concurrent processes. Defaults to the number of logical CPUs available. 
        Operates on a single process if given either 0 or 1
        For a small number of magnets, single-threading is dramatically faster due to the 
        overhead of spawning new processes. 
        
    
    '''
    if not isinstance(magnets, (list, tuple, np.ndarray)): # if a single magnet is passed to this program, then turn it into a list of magnets for simplicity. 
        magnets = [magnets]
    # The risk with multiprocessing is that completion is contingent on the slowest process
    # Typically this will be CoilPairs which act like a single magnet, but may contain 10s-100s
    # Therefore, if any CoilPairs are present, unwrap them into a single flat list containing the individual coils
    # This is not recursive, it assume there is only 1 level of wrapping, e.g. CoilPairs.
    # Offers about a 15-20% speedup on YQOMagnets
    unwrapped_magnets = []
    for magnet in magnets:
        for s in magnet:
            unwrapped_magnets.append(s)
    
    a1p, a2p, a3p = evaluate_axis_projection(projection) # This converts the projection into indicies. e.g. "zxy" will 
                                                         # put (rspace) z along the graph's x axis, (rspace) x along the 
                                                         # graph's y axis, and use a fixed (realspace) y value    
    axOneLim = limit
    axTwoLim = limit
    axOne = np.linspace(-axOneLim, axOneLim, points) + centre[a1p] # graph x axis
    axTwo = np.linspace(-axTwoLim, axTwoLim, points) + centre[a2p] # graph y axis
    X = np.outer(axOne, np.ones(points)) # co-ordinates of vector arrows along the (graph) x axis
    Y = np.outer(np.ones(points), axTwo) # and along the (graph) y axis
    U = np.zeros(X.shape) # Vector length along (Graph) x-axis
    V = np.zeros(X.shape) # and along graph (y-axis)
    C = np.zeros(X.shape) # Vector colour
    positions = [] # This list and the following while loop builds the list at which the B field will be evaluated. 
    for i, a1 in enumerate(axOne): # i and j are indicies for the location of the rspace co-ordinate within the grids X and Y
        for j, a2 in enumerate(axTwo):
            for m in unwrapped_magnets: # Each magnet is passed in separately, since if the worker has to iterate over the list, it runs into the GIL limitation
                coord = np.zeros(3)
                coord[a1p] = a1
                coord[a2p] = a2
                coord[a3p] = centre[a3p]
                argument = [m, coord, i, j] # i and j are passed to the function (and passed back) so that we do not rely on synchronous threading
                positions.append(argument)
    if not threads or threads == 1:
        print("Single threaded")
        out = [plot_vector_B_field_worker(argument) for argument in positions]
    else:
        # no point in using more processes than jobs:
        print("Multiprocessing with {} processes".format(threads))
        pool = multiprocessing.Pool(threads)
        out = pool.map(plot_vector_B_field_worker, positions)
        pool.close()
        pool.join()
    for element in out:
        i = element[1] # recover the i and j indicies to locate this actual value in the grid
        j = element[2]
        B_field = element[0]
        U[i, j] += B_field[a1p]
        V[i, j] += B_field[a2p]
        C[i, j] += np.linalg.norm(B_field)
    axes.quiver(X, Y, U, V, C, alpha=0.5, pivot='mid', angles='xy')
    axes.set_aspect('equal', 'datalim')
    axes.set_xlabel("%s [m]" % 'xyz'[a1p])
    axes.set_ylabel("%s [m]" % 'xyz'[a2p])
    axes.set_title(projection)
    return U, V






def plot_vector_B_field_worker(argument):
    '''argument is a list. arg[0] is a list of magnets, arg[1] is a co-ordinate in r-space at which to evaluate the field
    arg[2] and arg[3] are the (graph) x and y indicies of the value, to simplify reconstructing the grid after multiprocessing'''
    magnet = argument[0]
    coord = argument[1]
    i = argument[2]
    j = argument[3]
    B_field = np.zeros(3)
    B_field += magnet.get_B_field(coord)
    return B_field, i, j





def print_field_gradient(magnets, centre, label=""):
    ''' Print the value of the B-field gradient along the x and y axes around centre in G/cm'''
    delta = 1e-2
    B0 = np.zeros(3)
    B1 = np.zeros(3)
    B2 = np.zeros(3)
    r0 = centre
    rx = (centre[0] + delta, centre[1], centre[2])
    ry = (centre[0], centre[1]+delta, centre[2])
    for m in magnets:
        B0 += m.get_B_field(r0)
        B1 += m.get_B_field(rx)
        B2 += m.get_B_field(ry)
    B0abs = np.linalg.norm(B0)
    B1abs = np.linalg.norm(B1)
    B2abs = np.linalg.norm(B2)
    print("%s: x axis gradient: %.3g G/cm" % (label, ((B1abs - B0abs)*1e4 / (delta/1e-2)) )) # The factor multiplication converts from an absolute difference in Tesla to a gradient in G/cm
    print("%s: y axis gradient: %.3g G/cm" % (label, ((B2abs - B0abs)*1e4 / (delta/1e-2)) ))