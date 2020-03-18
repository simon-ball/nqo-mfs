# -*- coding: utf-8 -*-
"""
This file implements magnetic field calculations arising from several possible magnetic sources
You can create a combination of permanent magnets, straight wires and circular or rectangular coils (in either Helmholtz or Anti-helmholtz configuration).


Each magnet is created in its own co-ordinate system, r'=(x', y', z'), which is rotated from the laboratory frame by two angles, theta (rotation around Z) and phi (rotation around X).

Functions are provided for visualising the position of each magnet in various projections. 
"""

import numpy as np
from numpy import cos, sin, sqrt, power
from scipy.special import ellipk, ellipe
import random

from . import helpers

'''Constants'''
pi = np.pi
mu0 = 4*pi*1e-7


class Magnet:
    '''A representation of a generic magnetic field source (e.g. permanent magnet, solenoid, etc). The magnet is defined in its own co-ordinate axis r'=(x',y',z'),
    rotated with respect to the lab frame by theta and phi. 
    
    The angular convention used here is as follows:
        * Theta is the azimuthal angle
        * Phi is the polar angle
        
        This means that:
            If Phi=0°, values of Theta correspond to a rotation of the XY plane around the Z axis. 
            At Theta=90°, Phi=0°, the positive xDash axis is aligned along the positive Y axis
            
            If Theta=0°, values of Phi correspond to a rotation of the YZ plane around the X axis
            At Theta=0°, Phi=90°, the positive zDash axis is aligned along the positive Y axis
    
    Specific kinds of magnets extend this class to calculate the exact form of the magnetic field arising. '''
    
    def __init__(self, rDash, dimsDash, theta, phi=0):
        '''Initialise the generic magnet.
        
        Parameters
        ----------
        rDash : list
            The origin of the magnet in the Dashed co-ordinate system
        dimsDash : dict
            Dictionary of parameters. The exact parameters accepted are wide ranging, and are listed under the major, user-facing classes
        theta : float
            Rotation of the xDash-yDash axis around the Z axis in degrees
        phi : float
            Rotation of the yDash-zDash axis around the X axis in degrees
        
        Returns
        -------
        null
        '''
        self.theta = np.radians(theta)
        self.phi = np.radians(phi)
        self.rDash = rDash
        self.dimsDash = dimsDash.copy()
        self.fmat = random.sample(['b', 'g', 'r', 'c', 'm', 'y', 'k'], 1)[0] + '-'
        # This is a bit of a fudge - pick a colour for use in plotting this magnet in future.
        # THis is relevant where a single magnet may be plotted as several separate lines in matplotlib
        # i.e. either a rectangular PermanentMagnet or a CoilPair
        pass
    
    
    
    def rotate_to_dashed_frame(self, r):
        '''Angular rotation from the lab frame to the Dashed co-ordinate system

        Parameters
        ----------
        r : np.ndarray
            A vector in the lab frame : r = (x, y, z)
        
        Returns
        -------
        np.ndarray
            The vector in the Dashed co-ordinate system : rDash = (xDash, yDash, zDash)
        '''
        return helpers.rotate_to_dashed_frame(r, self.theta, self.phi)
        
    
    
    def rotate_to_normal_frame(self, rDash):
        '''Angular rotation from the  Dashed co-ordinate system to the lab frame

        Parameters
        ----------
        r : np.ndarray
            A vector in the Dashed co-ordinate system : rDash = (xDash, yDash, zDash)
        
        Returns
        -------
        np.ndarray
            The vector in the lab frame : r = (x, y, z)
        '''
        return helpers.rotate_to_normal_frame(rDash, self.theta, self.phi)
    
    
    
    def get_B_field(self, r):
        '''Calculate the vector B field at a given position r in the lab frame
        B(r) = B(x, y, z)

        Parameters
        ----------
        r : np.ndarray
            A vector in the lab frame : r = (x, y, z)
        
        Returns
        -------
        np.ndarray
            The vector B field at position r in the laboratory frame
        '''
        rDash = self.rotate_to_dashed_frame(r)
        BDash = self.get_BDash_field(rDash)        
        B = self.rotate_to_normal_frame(BDash)
        return B
    
    
    
    def get_BDash_field(self, rDash):
        '''Calculate the vector B field at a given position rDash in the Dashed co-ordinate system
        B(rDash) = B(xDash, yDash, zDash)
        
        This function is geometry specific and must be implemented in the subclass

        Parameters
        ----------
        rDash : np.ndarray
            A vector in the Dashed co-ordinate system : rDash = (xDash, yDash, zDash)
        
        Returns
        -------
        np.ndarray
            The vector B field at position rDash in the Dashed co-ordinate system
        '''
        raise NotImplementedError("Must be implemented in geometry-specific class")n
    
    
    
    def plot_magnet_position(self, axes, projection):
        '''Plot the outline of the magnet on the provided axes
        
        This only plots the 2D outline - e.g. an arbitrarily rotated cuboid will appear to be hexagonal
        This function is geometry specific and must be implemented in the subclass
        
        Parameters
        ----------
        axes : matplotlib.axes._subplots.AxesSubplot
            The pyplot axis on which the outline will be drawn. Can be produced by, e.g.
                fig, axes = plt.subplots()
                
        projection : str
            The plane into which the outline is being projected
            Accepted values are 2 letters out of 'xyz', with the first letter
            giving the real space axis of the graph's x-axis
            
        Returns
        -------
        null
        '''
        raise NotImplementedError("Must be implemented in geometry-specific class")n




class CircularCoil(Magnet):
    '''A representation of a single circular EM coil. The coil is defined as having its origin at rDash=(x',y',z'), and its axis along the yDash axis. 
    A positive current flowing results in a field that, at the coil's origin, points in the positive yDash direction. 
    This class represents a single loop of wire. For an approximation of multiple loops, multiply the current passing through this single loop. For higher precision,
    use the CircularCoilSpatial class, which extends this class to implement multiple coils at the correct spacing
    
    dimsDash is a dictionary which lists parameters related to the coil. The only relevant keyword for the CircularCoil class is "radius"'''
    def __init__(self, I, rDash, dimsDash, theta, phi=0):
        super(CircularCoil, self).__init__(rDash, dimsDash, theta, phi)
        self.I = I
        self.write_magnet_limits()
        pass
        
    def write_magnet_limits(self):
        '''The round coil is parameterised as follows via a dictionary (dimsDash)
        inner_radius, radial turn number, radial turn spacing, axial turn number, axial turn spacing'''
        self.radius = self.dimsDash['radius']
        n = 36 # Convert the circle into a polygon with this number of points
        ang = np.linspace(0,2*pi,n)
        xDash = (self.radius*sin(ang)) + self.rDash[0]
        zDash = (self.radius*cos(ang)) + self.rDash[2]
        yDash = self.rDash[1]
        coordinatesDash = np.array([xDash, yDash, zDash])
        self.coordinates = self.rotate_to_normal_frame(coordinatesDash)
        pass
    
    def get_BDash_field(self, rDash):
        '''Get the (x', y', z') components of the B field at position rDash=(x',y',z') for a single round loop
                   
        NB: this doc string should be augmented with a reference to the source of this calculation'''
        # First in polar co-ordinates
        xDash = rDash[0] - self.rDash[0]
        yDash = rDash[1] - self.rDash[1]
        zDash = rDash[2] - self.rDash[2] # These subtractions allow us to calculate as if the coil has its origin at (0,0,0)
        r = np.sqrt(xDash**2 + zDash**2) # radial distance of rDash from coil centre
                                         # !!NOTE!! within this function, r is treated as a **radius** and not as a vector position
        
        # Axial term, i.e. along yDash axis
        prefactor1 =  mu0 * self.I / (2*pi) 
        prefactor2 = 1. / np.sqrt( np.power(self.radius+r, 2) + np.power(yDash,2) )
        elliptic_argument = 4 * self.radius * r / (np.power(self.radius+r,2) + np.power(yDash,2) )
        term1 = ellipk(elliptic_argument)
        prefactor_term2 = (self.radius**2 - r**2 - np.power(yDash,2)) / (np.power(self.radius-r,2)+np.power(yDash,2))
        term2 = ellipe(elliptic_argument)
        
        B_axial = prefactor1 * prefactor2 * (term1 + (prefactor_term2*term2))

        #radial term, i.e. in the xDash-zDash plane
        if r == 0:
            B_radial=0
        else:
            prefactor3 = prefactor1 / r
            prefactor4 = prefactor2 * yDash
            term3 = -1*term1
            prefactor_term4 = (self.radius**2 + r**2 + np.power(yDash,2)) / (np.power(self.radius-r,2)+np.power(yDash,2))
            term4 = term2
            
            B_radial = prefactor3 * prefactor4 * (term3 + (prefactor_term4*term4))
            
        # Convert to cartesian co-ordinates        
        ByDash = B_axial
        if r == 0:
            BxDash = 0
            BzDash = 0
        else:
            BxDash = (xDash/r) * B_radial
            BzDash = (zDash/r) * B_radial
        return np.array([BxDash, ByDash, BzDash])

    def plot_magnet_position(self, axes, projection):
        '''Plot the outline of the magnet on the provided axes
        
        This only plots the 2D outline - e.g. an arbitrarily rotated cuboid will appear to be hexagonal
        This function is geometry specific and must be implemented in the subclass
        
        Parameters
        ----------
        axes : matplotlib.axes._subplots.AxesSubplot
            The pyplot axis on which the outline will be drawn. Can be produced by, e.g.
                fig, axes = plt.subplots()
                
        projection : str
            The plane into which the outline is being projected
            Accepted values are 2 letters out of 'xyz', with the first letter
            giving the real space axis of the graph's x-axis
            
        Returns
        -------
        null
        '''
        a1p, a2p, a3p = helpers.evaluate_axis_projection(projection)
        axes.plot(self.coordinates[a1p], self.coordinates[a2p], self.fmat)
        pass
     
        
class RectangularCoil(Magnet):
    '''A representation of a single rectangular EM coil. The coil is defined as having its origin at rDash=(x',y',z') and its axis along the yDash axis. 
    A positive current flowing results in a field which, at the coil origin, points along the positive yDash direction.
    
    Parameters:
            I: float. Current flowing through coil
            rDash: 3 element list or array. Position of origin of coil
            dimsDash: dictionary indicating dimensions of coil.
                Keywords for rectangular coil:
                    'axDash' : FULL length of coil along the xDash axis
                    'azDash' : FULL length of coil along the zDash axis
    '''
    def __init__(self, I, rDash, dimsDash, theta, phi):
        super(RectangularCoil, self).__init__(rDash, dimsDash, theta, phi)
        self.I = I
        self.write_magnet_limits()
        
    def write_magnet_limits(self):
        self.axD = self.dimsDash['axDash']/2
        self.azD = self.dimsDash['azDash']/2
        self.corners = np.zeros((5,3))
        a = (1,1,-1,-1, 1)
        b = (1,-1, -1, 1, 1)
        for i in range(5):
            c = [self.rDash[0] + (a[i]*self.axD), self.rDash[1], self.rDash[2] - (b[i]*self.azD)]
            self.corners[i,:] = self.rotate_to_normal_frame(np.array(c))

    def get_BDash_field(self, rDash):
        '''Approach based on Misiaken from  J. Res. Natl. Inst. Stand. Technol. 105, 557 (2000). I make two changes:
            * consistent with the other sources in this file, to choose yDash as the principle axis, and not Z
            * Due to Python zero-indexing arrays, the use of (-1)**alpha changes slightly'''
        prefactor = mu0 * self.I / (4*pi)
        xD = rDash[0] - self.rDash[0]
        yD = rDash[1] - self.rDash[1]
        zD = rDash[2] - self.rDash[2]
        
        # Abbreviations from paper:
        C1 = self.axD + xD
        C2 = self.axD - xD
        C3 = -C2
        C4 = -C1
        C = [C1, C2, C3, C4]
        d1 = zD + self.azD
        d2 = d1
        d3 = zD - self.azD
        d4 = d3
        d = [d1, d2, d3, d4]
        t1 = sqrt( power(self.axD + xD, 2) + power(self.azD + zD, 2) + power(yD,2))
        t2 = sqrt( power(self.axD - xD, 2) + power(self.azD + zD, 2) + power(yD,2))
        t3 = sqrt( power(self.axD - xD, 2) + power(self.azD - zD, 2) + power(yD,2))
        t4 = sqrt( power(self.axD + xD, 2) + power(self.azD - zD, 2) + power(yD,2))
        t = [t1, t2, t3, t4]
        # Compared to Misiakin, I have used t instead of r here, since r is somewhat overused already. 
        
        BxDash = 0
        ByDash = 0
        BzDash = 0
        for alpha in range(4): 
            # In comparison to Misiakin, here I use a zero-referenced sum, instead of a 1-refernced sum. 
            # Therefore, where Misiakin has (-1)**alpha+1, I use Q=(-1)**alpha
            # And in the one case where Misiakin has (-1)**alpha (only in his eq. 4), I use P=(-1)**(alpha+1)
            Q = (-1) ** alpha
            P = (-1) ** (alpha+1)
            BxDash += (Q * yD) / (t[alpha] * (t[alpha]*d[alpha]))
            BzDash += (Q*yD) / (t[alpha] * (t[alpha] + (Q*C[alpha])))
            ByDash_term1 = (P * d[alpha]) / (t[alpha] * (t[alpha] + (Q*C[alpha])))
            ByDash_term2 = -C[alpha] / (t[alpha] * (t[alpha] + d[alpha]))
            ByDash += ByDash_term1 + ByDash_term2
        BDash = prefactor * np.array( [BxDash, ByDash, BzDash] )
        return BDash
    
    def plot_magnet_position(self, axes, projection):
        '''Plot the outline of the magnet on the provided axes
        
        This only plots the 2D outline - e.g. an arbitrarily rotated cuboid will appear to be hexagonal
        This function is geometry specific and must be implemented in the subclass
        
        Parameters
        ----------
        axes : matplotlib.axes._subplots.AxesSubplot
            The pyplot axis on which the outline will be drawn. Can be produced by, e.g.
                fig, axes = plt.subplots()
                
        projection : str
            The plane into which the outline is being projected
            Accepted values are 2 letters out of 'xyz', with the first letter
            giving the real space axis of the graph's x-axis
            
        Returns
        -------
        null
        '''
        a1p, a2p, a3p = helpers.evaluate_axis_projection(projection)
        axes.plot(self.corners[:,a1p], self.corners[:,a2p], self.fmat)
        pass




class PermanentMagnet(Magnet):
    '''A representation of a rectangular permanent magnet. The magnet is defined in its own co-ordinate system, (x', y', z'),
    with the centre of the magnet at rDash and magnetisation M aligned parallel to the y' axis. 
    
    dimsDash is a dictionary containing the keys axDash, ayDash, azDash. These are the HALF lengths of the magnet in the x'd, y', z' directions'''
    def __init__(self, M, rDash, dimsDash, theta, phi=0):
        '''Initialise a permanent magnet
        
        Parameters
        ----------
        M : float
            The magnetisation of the magnet, in Tesla per metre
        rDash : list
            The origin of the magnet in the Dashed co-ordinate system
        dimsDash : dict
            Dictionary of parameters. The required parameters for a permanent magnet are:
                axDash
                ayDash
                azDash
            All three parameters are the FULL lengths of the magnet, in the x', y', z' frame, in metres
        theta : float
            Rotation of the xDash-yDash axis around the Z axis in degrees
        phi : float
            Rotation of the yDash-zDash axis around the X axis in degrees
        
        Returns
        -------
        null
        '''
        super(PermanentMagnet, self).__init__(rDash, dimsDash, theta, phi)
        self.M = M
        self.write_magnet_limits()        
        pass
    
    def write_magnet_limits(self):
        '''Calculate the position of the magnet verticies in the dash coordinate system.
        
        This function is a bit of a hodge-podge, because one format of verticies 
        is easier for B-field calculation, and an entirely different format is 
        useful for plotting the geomtry
        
        Part 1 calculates three 2-element arrays - these are the limits in the 
        x', y', z' directions respectively
        
        Part 2 produces four arrays, each containing five 3-element positions. 
        Each of the four arrays corresponds to one face of the magnet, which 
        can then be plotted to show the geometry. The 2 remaining sides are 
        implicitly shown as all edges have then been plotted
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None'''
        # part 1: limits in format useful for Bfield calculation
        self.mxDash = np.array([ self.rDash[0] - self.dimsDash['axDash']/2, 
                                self.rDash[0] + self.dimsDash['axDash']/2] )
        self.myDash = np.array([ self.rDash[1] - self.dimsDash['ayDash']/2, 
                                self.rDash[1] + self.dimsDash['ayDash']/2] )
        self.mzDash = np.array([ self.rDash[2] - self.dimsDash['azDash']/2, 
                                self.rDash[2] + self.dimsDash['azDash']/2] )
        self.lyDash = self.dimsDash['ayDash']*2
        # Part 2: limits in format useful for plotting
        self.axD = self.dimsDash['axDash']/2  
        self.azD = self.dimsDash['azDash']/2 
        self.ayD = self.dimsDash['ayDash']/2 
        self.corners = np.zeros((5,3))
        a = (1,1,-1,-1, 1)
        b = (1,-1, -1, 1, 1)
        face1 = np.zeros((5,3))
        face2 = np.zeros((5,3))
        face3 = np.zeros((5,3))
        face4 = np.zeros((5,3))
        for i in range(5):
            c1 = [self.rDash[0] + (a[i]*self.axD), self.rDash[1] + self.ayD, self.rDash[2] - (b[i]*self.azD)]
            face1[i] = self.rotate_to_normal_frame(np.array(c1))  
            c2 = c1[:]
            c2[1] = self.rDash[1] - self.ayD
            face2[i] = self.rotate_to_normal_frame(np.array(c2))
        face3[0] = face1[0]
        face3[1] = face1[1]
        face3[2] = face2[1]
        face3[3] = face2[0]
        face3[4] = face1[0]
        face4[0] = face1[2]
        face4[1] = face1[3]
        face4[2] = face2[3]
        face4[3] = face2[2]
        face4[4] = face1[2]
        self.faces = [face1, face2, face3, face4]
        arrow_start = self.rotate_to_normal_frame(np.array([ self.rDash[0], self.rDash[1], self.rDash[2] ]) )
        arrow_stop = self.rotate_to_normal_frame(np.array([0, np.sign(self.M)*self.ayD, 0]))
        self.arrow = [arrow_start, arrow_stop]
        pass

    def plot_magnet_position(self, axes, projection):
        '''Intended to supersede the hardcoded projections'''
        a1p, a2p, a3p = helpers.evaluate_axis_projection(projection)        
        for face in self.faces:
            axes.plot(face[:,a1p], face[:,a2p], self.fmat)
        axes.arrow(self.arrow[0][a1p], self.arrow[0][a2p], self.arrow[1][a1p], self.arrow[1][a2p])
        pass

    def get_BDash_field(self, rDash):
        '''Calculate the vector B field resulting from a single permanent magnet 
        at a given position rDash in the Dashed co-ordinate system
        B(rDash) = B(xDash, yDash, zDash)
        
        This function implements the approach taken by Dreidemensionale_Feldberechnung_starr_magnetierter_permanentmagnete_mit_einer_anwendung_in_eisenlosen_elektrischen_maschinen.pdf
        (accessible on the wiki), choosing magnetisation to be along the y' axis

        Parameters
        ----------
        rDash : np.ndarray
            A vector in the Dashed co-ordinate system : rDash = (xDash, yDash, zDash)
        
        Returns
        -------
        np.ndarray
            The vector B field at position rDash in the Dashed co-ordinate system
        '''
        BxDash = 0
        ByDash = 0
        BzDash = 0
        for i in [0,1]:
            for j in [0,1]:
                for k in [0,1]:
                    Q = (-1)**(i+j+k)
                    xiDash = rDash[0] - self.mxDash[i]
                    yjDash = rDash[1] - self.myDash[j]
                    zkDash = rDash[2] - self.mzDash[k]
                    BxDash += (Q * np.arcsinh( zkDash / 
                                             np.sqrt( np.power(xiDash, 2) + np.power(yjDash, 2) )
                                                     ))
                    BzDash += (Q * np.arcsinh( (rDash[0] - self.mxDash[k]) / 
                                             np.sqrt( np.power(rDash[2] - self.mzDash[i], 2) + np.power(yjDash, 2) )
                                                     ))
                    ByDash += (Q * np.arctan( xiDash * zkDash / 
                                            ( yjDash * np.sqrt(np.power(xiDash, 2) + np.power(yjDash,2) + np.power(zkDash,2)) )
                                            ))
        BDash = (mu0 * self.M / (4*pi) ) * np.array( [-BxDash, ByDash, -BzDash] )
        return BDash





class CoilPair(Magnet):
    '''A generic pair of coils. They can both be circular or rectangular, and either Helmholtz or Anti-Helmholtz.
    PARAMETERS:
        I
            float. Current flowing through coils
        rDash
            3 entry list or array. Origin of coils in Dash co-ordinate system
        dimsDash
            Dictionary. Relevant parameters, including the following:
                'full spacing' - float - full spacing between closest coils along yDash axis
                'spatially distributed' - bool - Should the programme calculate for each loop independently, or approximate by placing all coils in the same place?
                'radius' - float - radius of round coils
                'axDash' - float - half length of rectangular coil along xDash axis
                'ayDash' - float - half length of rectangular coil along zDash axis
                'axial layers' - int - number of new layers further out along the yDash axis
                'radial layers' - int - number of new layers further out away from yDash axis
                'configuration' - str - Valid entries are 'ahh', 'hh' and some variations
                'shape' - str - Valid entries are variations on 'circ', 'circular', 'rectangular', 'r', etc
        theta
            float. Rotation of Dash system around Z axis
        phi
            float. Rotation of Dash system around X axis
        
        '''
    def __init__(self, I, rDash, dimsDash, theta, phi):
        super(CoilPair, self).__init__(rDash, dimsDash, theta, phi)
        self.theta = theta  # Explicitly overwrite the values stored by super.init
        self.phi = phi      #  - otherwise we convert to radians multiple times!
        self.I = I
        self.handle_text_arguments()
        self.create_magnets()
        
    def handle_text_arguments(self):
        if self.dimsDash['shape'].lower() in ['circ', 'circular', 'round', 'c', 'spherical', 'cylindrical', 'cylinder']:
            self.shape = CircularCoil
            self.dimsDash['axDash'] = 0 # create these as dummy values to simplify later handling. They will never be used, they just allow the code to skip some conditionals
            self.dimsDash['azDash'] = 0 # create these as dummy values to simplify later handling. They will never be used, they just allow the code to skip some conditionals
        elif self.dimsDash['shape'].lower() in ['rect', 'rectangular', 'r', 'square', 'cube', 'cubic']:
            self.shape = RectangularCoil
            self.dimsDash['radius'] = 0 # create these as dummy values to simplify later handling. They will never be used, they just allow the code to skip some conditionals
        else:
            raise ValueError('Error: shape not understood. Try either "circular" or "rectangular"')
            
        if self.dimsDash['configuration'].lower() in ['hh', 'helmholtz', 'helm-holtz']:
            self.conf = 1
            # Helmholtz
        elif self.dimsDash['configuration'].lower() in ['ahh', 'anti-helmholtz', 'antihelmholtz']:
            self.conf = -1
            # Antihelmholtz
        else:
            raise ValueError('Error: configuration not understood. Please give the configuration as a string in the form "ahh" or "hh"')
        pass
    def change_current(self, new_I):
        if not self.spatial: #using the approximation - have to allow for this when changing current
            new_I *= self.dimsDash['axial layers'] * self.dimsDash['radial layers']
        for mag in self.magnets:
            mag.I = new_I
        pass
    
    def create_magnets(self):
        self.half_spacing = self.dimsDash['full spacing'] / 2
        self.spatial = self.dimsDash['spatially distributed']
        self.magnets = []
        if not self.spatial:
            # Multiple turns are placed in exactly the same place. Faster to calculate, but less accurate. In code, this is implemented by 1 loop (per coil) with n*I current flowing
            n = self.dimsDash['axial layers'] * self.dimsDash['radial layers']
            upper_origin = [self.rDash[0], self.rDash[1] + self.half_spacing, self.rDash[2]]
            lower_origin = [self.rDash[0], self.rDash[1] - self.half_spacing, self.rDash[2]]
            upper_coil = self.shape(self.I*n, upper_origin, self.dimsDash, self.theta, self.phi) # For overlapped coils, simply implement 1 turn with N* higher current in it.
            lower_coil = self.shape(self.I*n*self.conf, lower_origin, self.dimsDash, self.theta, self.phi) # Invert current direction via self.conf for AHH coils
            self.magnets.append(upper_coil)
            self.magnets.append(lower_coil)
        else:
            # Multiple turns are distributed through space as appropriate. Each loop has the correct current flowing. 
            # Substantially slower to calculate, but more accurate. 
            # The axial spacing is implemented by changing the origin (given to the coil itself as rDash). 
            # The radial spacing is implemented by modifying the dimsDash dictionary passed
            # to the individual coils. Note that .copy() must be used here, otherwise all coils would use the 
            # *same* dictionary in memory, screwing things up
            yStacks = self.dimsDash['axial layers']
            radStacks = self.dimsDash['radial layers']
            ySpacing = self.dimsDash['axial spacing']
            radSpacing = self.dimsDash['radial spacing']
            for a in range(yStacks):
                for b in range(radStacks):
                    upper_origin = [self.rDash[0], self.rDash[1] + self.half_spacing + a*ySpacing, self.rDash[2]]
                    lower_origin = [self.rDash[0], self.rDash[1] - self.half_spacing - a*ySpacing, self.rDash[2]]
                    params = self.dimsDash.copy()
                    params['half spacing'] = self.half_spacing + a*ySpacing
                    params['axDash'] += b * radSpacing
                    params['radius'] += b * radSpacing
                    self.magnets.append(self.shape(self.I, upper_origin, params, self.theta, self.phi))
                    self.magnets.append(self.shape(self.I*self.conf, lower_origin, params, self.theta, self.phi))
        pass
                    
    def get_BDash_field(self, rDash):
        BDash = np.zeros(3)
        for m in self.magnets:
            BDash += m.get_BDash_field(rDash)
        return BDash

    def plot_magnet_position(self, axes, projection):
        '''Plot the outline of all magnets in this group on the provided axes with the provided projection
        
        This only plots the 2D outline - e.g. an arbitrarily rotated cuboid will appear to be hexagonal
        This function is geometry specific and must be implemented in the subclass
        
        Parameters
        ----------
        axes : matplotlib.axes._subplots.AxesSubplot
            The pyplot axis on which the outline will be drawn. Can be produced by, e.g.
                fig, axes = plt.subplots()
                
        projection : str
            The plane into which the outline is being projected
            Accepted values are 2 or 3 letters out of 'xyz', with the first letter
            giving the real space axis of the graph's x-axis, the second letter giving
            the real space axis of the graph's y-axis, and the third letter ignored. 
        
        Returns
        -------
        null
        '''
        for m in self.magnets:
            m.plot_magnet_position(axes, projection)
        pass
