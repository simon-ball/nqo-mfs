#! /bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:29:47 2018

@author: swball


This file implements magnetic field calculations for the NQO Ytterbium experiment at SDU"""

import numpy as np
import matplotlib.pyplot as plt
from time import time

import mfs




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

def print_field_at_3DMOT(magnets, distance=0.3):
    ''' Print the value of the B-field, and the B-field gradient, at the position of the 3D MOT. 
    BY convention, the 3D MOT is located in the positive Z direction at a separation from the centre of the 2D MOT given by distance'''
    rMOT = (0,0,distance)
    delta = 0.01
    rMOTGrad = (delta, 0, distance)
    BMOT = np.zeros(3)
    BMOTGrad = np.zeros(3)
    for m in magnets:
        BMOT += m.get_B_field(rMOT)
        BMOTGrad += m.get_B_field(rMOTGrad)
    BMOT *= 1e7 # convert to milliGauss
    BMOTGrad *= 1e4 * (delta/0.01) # convert to Gauss/cm
    print("Field at MOT position (%d cm in +z): (xyz) %.3g, %.3g, %.3g mG" % (distance*100, BMOT[0], BMOT[1], BMOT[2]))
    print("Field gradient at MOT position (%d cm in +z) (x): %.3g G/cm" % (distance*100,np.linalg.norm(BMOTGrad)))
    
def print_field_at_position(magnets, position, label):
    '''Print the value of the B field at the table surface. This is important to determine how much protection the cell requires from, e.g., bolts and washers that will be left around during construction'''
    rTable = position
    BTable = np.zeros(3)
    if type(magnets) != list:
        magnets = [magnets]
    for m in magnets:
        BTable += m.get_B_field(rTable)
    BTable *= 1e4 # convert to Gauss
    print("%s: Absolute field at location %s: %.3g G" % (label, str(position), np.linalg.norm(BTable)))
    
def plot_field_gradient_for_Z(magnets, axes, zl, zh):
    ''' Plot the value of the B-field along the X axis at y=0 as a function of position on the Z axis'''
    points = 101
    axis = np.linspace(zl, zh, points)
    B_grad = np.zeros(points)
    for a, zpos in enumerate(axis):
        r0 = [0,0,zpos]
        r1 = [0.01, 0, zpos]
        B_field_0 = np.zeros(3)
        B_field_x = np.zeros(3)
        for m in magnets:
            B_field_0 += m.get_B_field(r0)
            B_field_x += m.get_B_field(r1)
        B_grad[a] += np.linalg.norm(B_field_x) - np.linalg.norm(B_field_0) # this is a difference in Tesla over a distane of 1cm
    B_grad_norm = B_grad / np.max(B_grad)
    axes.plot(axis*100, B_grad_norm)
    axes.set_xlabel("Z position [cm]")
    axes.set_ylabel("Grad B [norm]")
    axes.set_title("XY field gradient as distance along Z")
     
    


def plot_B_field_gradient_X(magnets, axes):
    ''' Plot the absolute value of the B-field as a function of position along the X axis at y,z=0 '''
    xLim = 0.04
    points = 201
    axis = np.linspace(-xLim, xLim, points)
    B_field_x= np.zeros([points, 3])
    B_field_y= np.zeros([points, 3])
    for a, pos in enumerate(axis):
        for magnet in magnets:
            rx = [pos, 0, 0]
            ry = [0,pos, 0]
            B_field_x[a, :] += magnet.get_B_field(rx)
            B_field_y[a, :] += magnet.get_B_field(ry)
    B_field_x_abs = np.linalg.norm(B_field_x, axis=1)
    B_field_y_abs = np.linalg.norm(B_field_y, axis=1)
    axes.plot(axis*1e2, B_field_x_abs*1e4, label="x axis")
    axes.plot(axis*1e2, B_field_y_abs*1e4, label="y axis")
    axes.set_xlabel("Position [cm]")
    axes.set_ylabel("B field magnitude [G]")
    axes.set_title("Field gradient in XY plane")
    axes.legend()
    

    
def plot_diagonal_field_pointing(axes, X, Z):
    '''This takes the calculated output of the plot_vector_B_field routine to do a quick plot of the effect of the 2D magnets on the 3D qudrupole field.
    To work, it requires that the aforementioned method calculates on a square grid (i.e. axOneLim == axTwoLim)'''
    p = X.shape[0]
    lim = 0.055
    diag = np.zeros((2,p))
    for i in range(p):
        diag[0,i] = X[i,i]
        diag[1,i] = Z[i,i]
    pointing = diag[0,:] / diag[1,:]
    axes.plot(np.linspace(-lim, lim, p), pointing)
    axes.set_xlabel("$\sqrt{x^2 + (z-z_0)^2}$ [m]")
    axes.set_ylabel("Vector aspect ratio")





if __name__ == "__main__":
    plt.close('all')
    ''' For the YQO magnetic field setup, we make the following choices:
        * the atomic beam propagates along the +ve direction along the Z-axis
        * the Y-axis is vertical
        * The X-axis therefore corresponds to the probe beam directions
        * The origin is at the centre of the 3D MOT coils, also the location of the probe focus
        * The 2D MOT origin is taken as the centre point of the 'Pancake' of the glass cell
        
        In addition, all units are SI units unless otherwise stated (metres, Tesla, Amps, etc)'''

 
    
    '''MOT coils'''
    MOTCoilOrigin = np.array([0,0,0])
    MOTCoilDimensions = {'radius':0.118, 'full spacing': 0.1, 'spatially distributed':True, 
                'axial layers': 8, 'radial layers':8, 'axial spacing':0.0041, 
                'radial spacing':0.0041, 'configuration':'ahh', 'shape':'circ'}
    MOTCoilTheta = 0
    MOTCoilPhi = 0
    MOTCoilCurrent = 25
    MOTCoil = [mfs.sources.CoilPair(MOTCoilCurrent, MOTCoilOrigin, MOTCoilDimensions, MOTCoilTheta, MOTCoilPhi)]
    
    
    
    '''MOT Bias coils'''
    #Around the probe axis
    MOTBiasCoilProbeDimensions = {'axDash': 0.065, 'azDash':0.268, 'full spacing':0.215,
                             'spatially distributed':True, 'axial layers':7, 'radial layers': 7,
                             'axial spacing':0.001, 'radial spacing':0.001, 'configuration':'hh', 'shape':'rect'}
    #Around the atom beam axis
    MOTBiasCoilAtomBeamDimensions = {'axDash': 0.08, 'azDash':0.08, 'full spacing':0.328,
                             'spatially distributed':True, 'axial layers':7, 'radial layers': 7,
                             'axial spacing':0.001, 'radial spacing':0.001, 'configuration':'hh', 'shape':'rect'}
    #Around the vertical axis
    MOTBiasCoilVerticalDimensions = {'axDash': 0.08, 'azDash':0.08, 'full spacing':0.328,
                             'spatially distributed':True, 'axial layers':7, 'radial layers': 7,
                             'axial spacing':0.001, 'radial spacing':0.001, 'configuration':'hh', 'shape':'rect'}
    MOTBiasCoilCurrent = 1
    MOTBiasCoilProbe = mfs.sources.CoilPair(MOTBiasCoilCurrent, MOTCoilOrigin, MOTBiasCoilProbeDimensions, theta=90, phi=0) # rotate such that Y'dash is along the X-axis=probe axis
    MOTBiasCoilAtomBeam = mfs.sources.CoilPair(MOTBiasCoilCurrent, MOTCoilOrigin, MOTBiasCoilAtomBeamDimensions, theta=0, phi=90) # rotate such that Y'dash is along the Z-axis = atom beam axis - NB, this requires that the 3D MOT be locate at [0,0,0]
    MOTBiasCoilVertical = mfs.sources.CoilPair(MOTBiasCoilCurrent, MOTCoilOrigin, MOTBiasCoilVerticalDimensions, theta=0, phi=0) # Do not rotate, Y'Dash= vertical. 
    
    
    MOTBiasCoils = [MOTBiasCoilProbe, MOTBiasCoilAtomBeam, MOTBiasCoilVertical]
    
    
    '''2D MOT permanent magnets'''
    twoDMotOrigin = [0, 0, -0.337] # Arbitrarily, this is set at the closer surface of the "pancake" of the 2dmot glass cell. From inventor, this is -337mm in Z
    magnetisation = 8.8e5 # A/m
    rDash0 = [0.055, 0, 0]
    theta = 45
    phi = 0
    xoff = 10e-3 # x' stacking offset - this doesn't affect the rotational stuff. 
    yoff = 3e-3 # y' stacking offset
    zoff = (25+4)*1e-3
    magnets2D = []
    xDashExtra = 0.00 # Use this to consider how the flatness changes if the magnets are placed further out to achieve a lower field gradient
    zDashR = [-0.008, 0.023, 0.053, 0.083, 0.113, -0.008, 0.135] # z-positions of magnet stacks relative to the 2D MOT origin (glass cell pancake surface)
    xDashR = [ 0.064, 0.055, 0.057, 0.056, 0.056, 0.079, 0.059]
    rotated = [0, 0, 0, 0, 0, 0, 1]
    for i in range(7):
        yran = [-1.5, -0.5, 0.5, 1.5]
        if not rotated[i]:
            dd = {'axDash':0.010, 'ayDash':0.003, 'azDash':0.025}
            
        else: 
            dd = {'azDash':0.010, 'ayDash':0.003, 'axDash':0.025}
            # Rotate the magnet around the yDash axis.
        for y in yran:
            rDashPos = np.array( [xDashR[i]+xDashExtra, y*yoff, zDashR[i]] ) + twoDMotOrigin
            rDashNeg = np.array( [-xDashR[i]-xDashExtra, y*yoff, zDashR[i]] ) + twoDMotOrigin          
            magnets2D.append(mfs.sources.PermanentMagnet(-magnetisation, rDashPos, dd, theta, phi))
            magnets2D.append(mfs.sources.PermanentMagnet(magnetisation, rDashNeg, dd, theta, phi))
            magnets2D.append(mfs.sources.PermanentMagnet(magnetisation, rDashPos, dd, theta-90, phi))
            magnets2D.append(mfs.sources.PermanentMagnet(-magnetisation, rDashNeg, dd, theta-90, phi))
    
    
    
    
    '''2D MOT bias coils'''
    # Vertical
    twoDMOTBiasOrigin = [0, 0, -0.269]
    twoDMOTBiasDimensions = {'axDash': 0.04, 'azDash':0.1, 'full spacing':0.092,
                             'spatially distributed':True, 'axial layers':10, 'radial layers': 9,
                             'axial spacing':0.001, 'radial spacing':0.001, 'configuration':'hh', 'shape':'rect'}
    twoDMOTBiasCurrent = 1
    twoDMOTBiasCoilsVertical = mfs.sources.CoilPair(twoDMOTBiasCurrent, twoDMOTBiasOrigin, twoDMOTBiasDimensions, theta=0, phi=0)
    twoDMOTBiasCoilsHorizontal = mfs.sources.CoilPair(twoDMOTBiasCurrent, twoDMOTBiasOrigin, twoDMOTBiasDimensions, theta=90, phi=0)
    
    twoDBiasCoils = [twoDMOTBiasCoilsVertical, twoDMOTBiasCoilsHorizontal]
    
    
    biasCoils = MOTBiasCoils + twoDBiasCoils
    quadrupoleMagnets = MOTCoil + magnets2D
    allMagnets = quadrupoleMagnets + biasCoils
    
    
    
    
    '''Field gradients: Quadrupolar magnets'''
    print("!!!           Quadrupolar magnetic sources           !!!")
    print_field_gradient(magnets2D, centre=twoDMotOrigin, label="2D MOT alone")
    print_field_gradient(MOTCoil, centre=MOTCoilOrigin, label="3D MOT alone")
    print('')
    
    '''Bias field scaling'''
    print('!!!     Bias field scaling: absolute field at 1A     !!!')
    print_field_at_position(twoDMOTBiasCoilsVertical, twoDMOTBiasOrigin, label='2D MOT bias coils')
    print_field_at_position(MOTBiasCoilVertical, MOTCoilOrigin, label='3D Vertical bias coils')
    print_field_at_position(MOTBiasCoilAtomBeam, MOTCoilOrigin, label='3D atom beam bias coils')
    print_field_at_position(MOTBiasCoilProbe, MOTCoilOrigin, label='3D probe bias coils')
    print('')
    
    
    
    
    '''Plot the geometric layout'''    
    fig1, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')
    for mag in allMagnets:       
        mag.plot_magnet_position(ax1, 'xy')
        mag.plot_magnet_position(ax2, 'zy')
        mag.plot_magnet_position(ax3, 'xz')
    fig1.tight_layout()
    
    






    thread_start = time()
    points = 50
    ncpu = 8
    
    mfs.plot_vector_B_field(magnets2D, ax1, centre=twoDMOTBiasOrigin, points=points, limit=0.03, projection='xy', threads=ncpu)
    mfs.plot_vector_B_field(magnets2D, ax2, centre=twoDMOTBiasOrigin, points=points, limit=0.04, projection='zy', threads=ncpu)
    mfs.plot_vector_B_field(MOTCoil, ax2, centre=MOTCoilOrigin, points=points, limit=0.07, projection='zy', threads=ncpu)
    mfs.plot_vector_B_field(MOTCoil, ax3, centre=MOTCoilOrigin, points=points, limit=0.07, projection='xz', threads=ncpu)
    
    thread_stop = time()
    print("Threaded time: %.2f" % (thread_stop - thread_start))

    

    
    '''Plot the behaviour of the field along the Z axis'''
    print_field_at_3DMOT(magnets2D, distance=0.336)
    fig2, (ax4, ax5) = plt.subplots(2,1)
    plot_field_gradient_for_Z(magnets2D, ax4, zl=-0.05, zh=0.4)
    plot_field_gradient_for_Z(magnets2D, ax5, zl=0.032, zh=0.112)
    
    fig2.tight_layout()

    
    