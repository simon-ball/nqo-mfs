import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from . import sources
from . import helpers

ncpu = multiprocessing.cpu_count()

def permanent_magnets():
    '''Create a pair of permanent magnets, some distance away from the origin
    
    The magnets are some distance apart in xDash, with magnetisation along (but pointing in opposite directions) yDash'''
    
    rDash0 = [0.1,0,0]
    rDash1 = [-0.1,0,0]
    theta = 45
    phi = 0
    magnetDimensions = {'axDash':0.05, 'ayDash':0.02, 'azDash':0.05}
    magnetisation = 1.0e6
    pMag0 = sources.PermanentMagnet(magnetisation, rDash0, magnetDimensions, theta, phi)
    pMag1 = sources.PermanentMagnet(magnetisation, rDash1, magnetDimensions, theta, phi)
    
    magnets = [pMag0, pMag1]
    
    '''Plot the geometry of these magnets in 3 projections'''
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    for mag in magnets:        
        mag.plot_magnet_position(ax1, 'xy')
        mag.plot_magnet_position(ax2, 'zy')
        mag.plot_magnet_position(ax3, 'xz')
    
    '''Plot the vector field of these magnets on the same graphs (and so, make sure to use the same projections!)'''
    '''Note! Axes 2 and 3 can be a little bit confusing - because you're plotted the 2D projection of the magnets
    and then overlaying that with the vector B-field in the zy, xz PLANES. The magnets are drawn AS IF THEY WERE IN 
    those planes, but they AREN'T, they are out of plane'''
    helpers.plot_vector_B_field(magnets, ax1, centre=[0,0,0], points=20, limit=0.05, projection='xy', threads=ncpu)
    helpers.plot_vector_B_field(magnets, ax2, centre=[0,0,0], points=20, limit=0.1, projection='zy', threads=ncpu)
    helpers.plot_vector_B_field(magnets, ax3, centre=[0,0,0], points=20, limit=0.1, projection='xz', threads=ncpu)
    plt.show()
    
def rectangular_coil_pair():
    ''' Create a coil pair using rectangular, helmholtz-configured coils. For 
    simplicity, make the overlapping assumption, i.e. many coils overlapped on 
    top of each other'''
    
    origin = [0,0,0] # The origin of the coil pair is located here  note, technically, this is already in the Dashed frame. In this case, it doesn't matter since both systems have [0,0,0] at the same plate
    theta = 0
    phi = 0
    current = 10
    magnetDimensions = {'axDash':0.1, 'azDash':0.2, 'full spacing': 0.15, 
                        'spatially distributed' : False, 'axial layers': 10, 'radial layers' : 2, # Note - we still have to define layers radialy/axially
                        'shape':'rect', 'configuration':'hh'}
    mag = sources.CoilPair(current, origin, magnetDimensions, theta, phi)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
          
    mag.plot_magnet_position(ax1, 'xy')
    mag.plot_magnet_position(ax2, 'zy')
    mag.plot_magnet_position(ax3, 'xz')
    helpers.plot_vector_B_field(mag, ax1, centre=[0,0,0], points=20, limit=0.05, projection='xy', threads=ncpu)
    helpers.plot_vector_B_field(mag, ax2, centre=[0,0,0], points=20, limit=0.1, projection='zy', threads=ncpu)
    plt.show()
    
    
    
    
def circular_coil_pair():
    ''' Create a coil pair using circular, anti-helmholtz-configured coils, at 
    some wierd angle. For a more accurate simulation, set the spatially 
    distributed flag. For extra fun, assume non-circular wire. Position the 
    coils at [1,2,3] in the laboratory frame'''
    
    origin = [1,2,3] 
    theta = 10
    phi = 30
    originDash = helpers.rotate_to_dashed_frame(np.array(origin), np.radians(theta), np.radians(phi)) # The magnet class expects an origin in the dashed frame; and the "naked" rotation functions expect angles in radians
    current = 10
    magnetDimensions = {'radius': 0.1, 'full spacing': 0.15, 
                        'spatially distributed' : True, 'axial layers': 3, 'radial layers' : 3, 'axial spacing':0.001, 'radial spacing':0.002,
                        'shape':'circ', 'configuration':'ahh'}
    mag = sources.CoilPair(current, originDash, magnetDimensions, theta, phi)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
          
    mag.plot_magnet_position(ax1, 'xy')
    mag.plot_magnet_position(ax2, 'zy')
    mag.plot_magnet_position(ax3, 'xz')
    helpers.plot_vector_B_field(mag, ax1, centre=origin, points=10, limit=0.1, projection='xy', threads=ncpu)
    helpers.plot_vector_B_field(mag, ax2, centre=origin, points=10, limit=0.1, projection='zy', threads=ncpu)
    helpers.plot_vector_B_field(mag, ax3, centre=origin, points=10, limit=0.1, projection='xz', threads=ncpu)
    plt.show()

    
    