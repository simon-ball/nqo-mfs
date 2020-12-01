# -*- coding: utf-8 -*-
"""
This file implements magnetic field calculations arising from several possible
magnetic sources. You can create a combination of permanent magnets, straight
wires and circular or rectangular coils (in either Helmholtz or Anti-helmholtz
configuration).


Each magnet is created in its own co-ordinate frame, r'=(x', y', z'), which is
rotated from the laboratory frame by two angles, theta (rotation around Z) and
phi (rotation around X).

Functions are provided for visualising the position of each magnet in various
projections. 
"""
import numpy as np
from numpy import cos, sin, sqrt, power
from scipy.special import ellipk, ellipe

from . import helpers
from .version import __version__


"""Constants"""
pi = np.pi
mu0 = 4 * pi * 1e-7

LINESTYLE = "solid"


class Magnet(object):
    """A representation of a generic magnetic field source (e.g. permanent magnet,
    solenoid etc). The magnet is defined in its own co-ordinate axis ``r'=(x',y',z')``,
    rotated with respect to the lab frame by theta and phi. 
    
    The angular convention used here is as follows
    
    * Theta is the azimuthal angle
    * Phi is the polar angle
        
    This means that
    
    * If Phi=0°, values of Theta correspond to a rotation of the XY plane
      around the Z axis. 
    * At Theta=90°, Phi=0°, the positive xDash axis is aligned along the
      positive Y axis
    * If Theta=0°, values of Phi correspond to a rotation of the YZ plane
      around the X axis
    * At Theta=0°, Phi=90°, the positive zDash axis is aligned along the
      positive Y axis
    
    Specific kinds of magnets extend this class to calculate the exact form of
    the magnetic field arising.
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        """Initialise the generic magnet.
        
        Parameters
        ----------
        strength: float
            Magnet strength. Meaning varies by magnet type
        rDash: list
            The origin of the magnet in the Dashed co-ordinate frame
        dimsDash: dict
            Dictionary of parameters. The exact parameters accepted are wide 
            anging, and are listed under the major, user-facing classes
        theta: float, optional
            Rotation of the xDash-yDash axis around the Z axis in degrees.
            Defaults to 0 
        phi: float, optional
            Rotation of the yDash-zDash axis around the X axis in degrees.
            Defaults to 0
        name: str, optional
            Human readable label to attach
        """
        self.strength = strength
        self.idx = 0
        self._theta = theta
        self._phi = phi
        self.rDash = np.array(rDash)
        self.dimsDash = dimsDash.copy()
        self.name = name
        self._colour = helpers._generate_random_colour()
        # This is a bit of a fudge - pick a colour for use in plotting this magnet in future.
        # This is relevant where a single magnet may be plotted as several separate lines in matplotlib
        # i.e. either a rectangular PermanentMagnet or a CoilPair
        return

    def __iter__(self):
        return iter((self,))

    def __next__(self):
        if self.idx == 0:
            self.idx += 1
            return self
        else:
            self.idx == 0
            raise StopIteration

    def __len__(self):
        return 1

    @property
    def colour(self):
        return self._colour

    def set_colour(self, c):
        self._colour = c

    @property
    def size(self):
        return 1

    @property
    def theta_deg(self):
        """Return Theta in degrees"""
        return self._theta

    @property
    def theta(self):
        """return Theta in radians"""
        return np.radians(self._theta)

    @theta.setter
    def theta(self, _theta):
        self._theta = _theta

    @property
    def phi_deg(self):
        """return phi in degrees"""
        return self._phi

    @property
    def phi(self):
        """return phi in radians"""
        return np.radians(self._phi)

    @phi.setter
    def phi(self, _phi):
        self._phi = _phi

    def rotate_to_dashed_frame(self, r):
        """Angular rotation from the lab frame to the Dashed co-ordinate frame

        Parameters
        ----------
        r: np.ndarray
            A vector in the lab frame: ``r = (x, y, z)``
        
        Returns
        -------
        np.ndarray
            The vector in the Dashed co-ordinate frame: ``rDash = (xDash, yDash, zDash)``
        """
        return helpers.rotate_to_dashed_frame(r, self.theta, self.phi)

    def rotate_to_normal_frame(self, rDash):
        """Angular rotation from the Dashed co-ordinate frame to the lab frame

        Parameters
        ----------
        r: np.ndarray
            A vector in the Dashed co-ordinate frame: ``rDash = (xDash, yDash, zDash)``
        
        Returns
        -------
        np.ndarray
            The vector in the lab frame: ``r = (x, y, z)``
        """
        return helpers.rotate_to_normal_frame(rDash, self.theta, self.phi)

    def get_B_field(self, r):
        """Calculate the vector B field at a given position r in the lab frame
        ``B(r) = B(x, y, z)``

        Parameters
        ----------
        r: np.ndarray
            A vector in the lab frame: ``r = (x, y, z)``
        
        Returns
        -------
        np.ndarray
            The vector B field at position ``r`` in the laboratory frame
        """
        rDash = self.rotate_to_dashed_frame(r)
        BDash = self.get_BDash_field(rDash)
        B = self.rotate_to_normal_frame(BDash)
        return B

    def get_BDash_field(self, rDash):
        """Calculate the vector B field at a given position rDash in the Dashed
        frame ``B(rDash) = B(xDash, yDash, zDash)``

        Parameters
        ----------
        rDash: np.ndarray
            A vector in the Dashed co-ordinate frame: ``rDash = (xDash, yDash, zDash)``
        
        Returns
        -------
        np.ndarray
            The vector B field at position ``rDash`` in the Dashed co-ordinate frame
        """
        raise NotImplementedError("Must be implemented in geometry-specific class")

    def plot_magnet_position(self, axes, projection):
        """Plot the outline of the magnet on the provided axes
        
        If the axes are 2D, it will only plot the outline of the magnet as
        projected by the `projection` keyword. For example, a coil in the XY
        plane will appear to be a line with the projection `xy`
        
        If the axes are 3d, then all 3 dimensions will be plotted. 
        
        Parameters
        ----------
        axes: matplotlib.axes._subplots.AxesSubplot
            The pyplot axis on which the outline will be drawn. Can be produced by, e.g.
                fig, axes = plt.subplots()
        projection: str
            The projection of the global frame onto the axes, in the form of a
            string such as ``xyz``. Maps the global frame dimension at position
            ``i`` in the string onto the axis dimension ``i``.
            For example, ``projection="zxy"`` will project
            
            * global dimension z onto axes dimension x
            * global dimension x onto axes dimension y
            * global dimension y onto axes projection z
        """
        # self.coordinates must be provided by the geometry-specific class
        a1p, a2p, a3p = helpers.evaluate_axis_projection(projection)
        try:
            axes.set_xlabel(projection[0])
            axes.set_ylabel(projection[1])
        except:
            pass
        plotting_args = {"color": self.colour, "linestyle": LINESTYLE}
        if helpers._get_axes_ndim(axes) == 2:
            axes.plot(self.coordinates[a1p], self.coordinates[a2p], **plotting_args)
        elif helpers._get_axes_ndim(axes) == 3:
            axes.plot(
                self.coordinates[a1p],
                self.coordinates[a2p],
                self.coordinates[a3p],
                **plotting_args,
            )
            axes.set_zlabel(projection[a3p])
        return

    def _to_dict(self):
        output = {
            "name": self.name if self.name is not None else "",
            "class": str(type(self)).split("'")[1].split(".")[-1],
            "strength": self.strength,
            "origin": self.rDash.tolist(),
            "dimensions": self.dimsDash,
            "theta": float(self.theta_deg),
            "phi": float(self.phi_deg),
            "mfs_version": __version__,
        }
        return output

    @classmethod
    def _from_dict(cls, dic):
        """
        Create magnet from a dictionary
        """
        m = cls(
            strength=dic["strength"],
            rDash=dic["origin"],
            dimsDash=dic["dimensions"],
            theta=dic["theta"],
            phi=dic["phi"],
            name=dic["name"],
        )
        return m


class MagnetGroup(Magnet):
    """A generic group of magnets
    
    Parameters
    ----------
    strength: float
        Current flowing through coils
    rDash: 3 entry list or array
        Origin of coils in Dash co-ordinate frame
    dimsDash: dict
        Dictionary of coil pair parameters.
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self.magnets = []
        return

    def __iter__(self):
        return iter(self.magnets)

    def __next__(self):
        self.idx += 1
        try:
            return self.magnets[self.idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration

    def __len__(self):
        return len(self.magnets)
    
    @property
    def colour(self):
        return self._colour

    def set_colour(self, c):
        self._colour = c
        for m in self.magnets:
            m.set_colour(c)

    @property
    def size(self):
        return len(self.magnets)

    def get_BDash_field(self, rDash):
        BDash = np.zeros(3)
        for m in self.magnets:
            BDash += m.get_BDash_field(rDash)
        return BDash

    def plot_magnet_position(self, axes, projection):
        for m in self.magnets:
            m.plot_magnet_position(axes, projection)
        return


class CircularCoil(Magnet):
    """A representation of a single circular EM coil. The coil is defined as having 
    its origin at rDash=(x',y',z') and its axis along the yDash axis. A positive
    current flowing results in a field which, at the coil origin, points along the
    positive yDash direction.
    
    This class mmodels a single loop of wire. For an approximation of multiple loops,
    increase the current. For an accurate simulation of a spatially distributed
    buncdle of wires, there exists a shortcut in the CoilPair class, but no shortcut
    for a spatially distributed single coil
    
    Reference
    ---------
    https://doi.org/10.1103/PhysRevA.35.1535
    
    Magnetostatic trapping fields for neutral atoms (Bergman)
    
    1987, PRA 35 1535
    
    Parameters
    ----------
    strength: float
        Current flowing through coil
    rDash: 3 element list or array
            Position of origin of coil
    dimsDash: dict
        Dictionary of parameters. Required parameters for a circular coil are
        
        * ``radius``: float
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self._write_magnet_limits()
        return

    def _write_magnet_limits(self):
        """Generate co-ordinates for the `plot_magnet_position`"""
        self.radius = self.dimsDash["radius"]
        n = 36  # Convert the circle into a polygon with this number of points
        ang = np.linspace(0, 2 * pi, n)
        xDash = (self.radius * sin(ang)) + self.rDash[0]
        zDash = (self.radius * cos(ang)) + self.rDash[2]
        yDash = [self.rDash[1] for angle in ang]
        coordinatesDash = np.array([xDash, yDash, zDash])
        self.coordinates = self.rotate_to_normal_frame(coordinatesDash)
        return

    def get_BDash_field(self, rDash):
        # First in polar co-ordinates
        xDash, yDash, zDash = rDash - self.rDash
        # These subtractions allow us to calculate as if the coil has its origin at (0,0,0)
        r = np.sqrt(xDash ** 2 + zDash ** 2)
        # radial distance of rDash from coil centre
        # !!NOTE!! within this function, r is treated as a **radius** and not as a vector position

        # Axial term, i.e. along yDash axis
        prefactor1 = mu0 * self.strength / (2 * pi)
        prefactor2 = 1.0 / np.sqrt(np.power(self.radius + r, 2) + np.power(yDash, 2))
        elliptic_argument = (
            4 * self.radius * r / (np.power(self.radius + r, 2) + np.power(yDash, 2))
        )
        term1 = ellipk(elliptic_argument)
        prefactor_term2 = (self.radius ** 2 - r ** 2 - np.power(yDash, 2)) / (
            np.power(self.radius - r, 2) + np.power(yDash, 2)
        )
        term2 = ellipe(elliptic_argument)

        B_axial = prefactor1 * prefactor2 * (term1 + (prefactor_term2 * term2))

        # radial term, i.e. in the xDash-zDash plane
        if r == 0:
            B_radial = 0
        else:
            prefactor3 = prefactor1 / r
            prefactor4 = prefactor2 * yDash
            term3 = -1 * term1
            prefactor_term4 = (self.radius ** 2 + r ** 2 + np.power(yDash, 2)) / (
                np.power(self.radius - r, 2) + np.power(yDash, 2)
            )
            term4 = term2

            B_radial = prefactor3 * prefactor4 * (term3 + (prefactor_term4 * term4))

        # Convert to cartesian co-ordinates
        ByDash = B_axial
        if r == 0:
            BxDash = 0
            BzDash = 0
        else:
            BxDash = (xDash / r) * B_radial
            BzDash = (zDash / r) * B_radial
        return np.array([BxDash, ByDash, BzDash])


class RectangularCoil(Magnet):
    """A representation of a single rectangular EM coil. The coil is defined as
    having its origin at rDash=(x',y',z') and its axis along the yDash axis. A
    positive current flowing results in a field which, at the coil origin,
    points along the positive yDash direction.
    
    References
    ----------
    https://dx.doi.org/10.6028%2Fjres.105.045
    
    Equations for the Magnetic Field produced by One or More Rectangular
    Loops of Wire in the Same Plane (Misiakin)
    
    J Res Natl Inst Stand Technol. 2000 Jul-Aug; 105(4): 557–564
    
    Two slight changes from the paper
    
    * consistent with the other sources in this file, to choose ``yDash`` as
      the principle axis, and not ``Z``
    * Due to Python zero-indexing arrays, the use of ``(-1)**alpha`` changes
      slightly
    
    Parameters
    ----------
    strength: float
        Current flowing through coil
    rDash: 3 element list or array
        Position of origin of coil in the coil frame
    dimsDash: dict
        Dictionary of parameters. Required parameters for a rectangular coil are
        
        * ``axDash``: float
          Full length of coil along ``xDash`` axis
        * ``azDash``: float
          Full length of coil along ``zDash`` axis
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self._write_magnet_limits()
        return

    def _write_magnet_limits(self):
        self.axD = self.dimsDash["axDash"] / 2
        self.azD = self.dimsDash["azDash"] / 2
        self.coordinates = np.zeros((3, 5))
        a = (1, 1, -1, -1, 1)
        b = (1, -1, -1, 1, 1)
        for i in range(5):
            c = [
                self.rDash[0] + (a[i] * self.axD),
                self.rDash[1],
                self.rDash[2] - (b[i] * self.azD),
            ]
            self.coordinates[:, i] = self.rotate_to_normal_frame(np.array(c))
        return

    def get_BDash_field(self, rDash):
        prefactor = mu0 * self.strength / (4 * pi)
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
        t1 = sqrt(power(self.axD + xD, 2) + power(self.azD + zD, 2) + power(yD, 2))
        t2 = sqrt(power(self.axD - xD, 2) + power(self.azD + zD, 2) + power(yD, 2))
        t3 = sqrt(power(self.axD - xD, 2) + power(self.azD - zD, 2) + power(yD, 2))
        t4 = sqrt(power(self.axD + xD, 2) + power(self.azD - zD, 2) + power(yD, 2))
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
            P = (-1) ** (alpha + 1)
            BxDash += (Q * yD) / (t[alpha] * (t[alpha] + d[alpha]))
            BzDash += (Q * yD) / (t[alpha] * (t[alpha] + (Q * C[alpha])))
            ByDash_term1 = (P * d[alpha]) / (t[alpha] * (t[alpha] + (Q * C[alpha])))
            ByDash_term2 = -C[alpha] / (t[alpha] * (t[alpha] + d[alpha]))
            ByDash += ByDash_term1 + ByDash_term2
        BDash = prefactor * np.array([BxDash, ByDash, BzDash])
        return BDash


class PermanentMagnet(Magnet):
    """A representation of a rectangular permanent magnet. The magnet is defined
    in its own co-ordinate system, ``(x', y', z')``, with the centre of the magnet
    at rDash and magnetisation M aligned parallel to the y' axis. 
    
    dimsDash is a dictionary containing the keys axDash, ayDash, azDash. These
    are the FULL lengths of the magnet in the x'd, y', z' directions
    
    References
    ----------
    https://link.springer.com/content/pdf/10.1007%2FBF01573988.pdf
    
    This function implements the approach taken by Metzger in Archiv fur
    Elektrotechnik 59 (1977) 229-242, with one change - selecting ``yDash`` as
    the axis of magnetisation
    
    Parameters
    ----------
    strength: float
        The magnetisation of the magnet, in Tesla per metre
    rDash: list
        The origin of the magnet in the Dashed co-ordinate frame
    dimsDash: dict
        Dictionary of parameters. The required parameters for a permanent magnet are
        
        * ``axDash``
        * ``ayDash``
        * ``azDash``
        All three parameters are the FULL lengths of the magnet, in the x', y', z' frame, in metres
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self._write_magnet_limits()
        pass

    def _write_magnet_limits(self):
        """Calculate the position of the magnet verticies in the dash coordinate frame.
        
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
        None
        """
        # part 1: limits in format useful for Bfield calculation
        self.mxDash = np.array(
            [
                self.rDash[0] - self.dimsDash["axDash"] / 2,
                self.rDash[0] + self.dimsDash["axDash"] / 2,
            ]
        )
        self.myDash = np.array(
            [
                self.rDash[1] - self.dimsDash["ayDash"] / 2,
                self.rDash[1] + self.dimsDash["ayDash"] / 2,
            ]
        )
        self.mzDash = np.array(
            [
                self.rDash[2] - self.dimsDash["azDash"] / 2,
                self.rDash[2] + self.dimsDash["azDash"] / 2,
            ]
        )
        self.lyDash = self.dimsDash["ayDash"] * 2
        # Part 2: limits in format useful for plotting
        self.axD = self.dimsDash["axDash"] / 2
        self.azD = self.dimsDash["azDash"] / 2
        self.ayD = self.dimsDash["ayDash"] / 2
        self.corners = np.zeros((5, 3))
        a = (1, 1, -1, -1, 1)
        b = (1, -1, -1, 1, 1)
        face1 = np.zeros((5, 3))
        face2 = face1.copy()
        face3 = face1.copy()
        face4 = face1.copy()
        for i in range(5):
            c1 = [
                self.rDash[0] + (a[i] * self.axD),
                self.rDash[1] + self.ayD,
                self.rDash[2] - (b[i] * self.azD),
            ]
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
        self.faces = [face1.T, face2.T, face3.T, face4.T]
        # Easier to write the `faces` this way
        # Easier to read from them once transposed
        arrow_start = self.rotate_to_normal_frame(
            np.array([self.rDash[0], self.rDash[1], self.rDash[2]])
        )
        arrow_stop = self.rotate_to_normal_frame(
            np.array([0, np.sign(self.strength) * self.ayD, 0])
        )
        self.arrow = [arrow_start, arrow_stop]
        pass

    def plot_magnet_position(self, axes, projection):
        a1p, a2p, a3p = helpers.evaluate_axis_projection(projection)
        ndim = helpers._get_axes_ndim(axes)
        plotting_args = {"color": self.colour, "linestyle": LINESTYLE}
        if ndim == 2:
            for face in self.faces:
                axes.plot(face[a1p], face[a2p], **plotting_args)
            axes.arrow(
                self.arrow[0][a1p],
                self.arrow[0][a2p],
                self.arrow[1][a1p],
                self.arrow[1][a2p],
            )
        elif ndim == 3:
            for face in self.faces:
                axes.plot(face[a1p], face[a2p], face[a3p], **plotting_args)
            # arrows not currently supported in 3D
        return

    def get_BDash_field(self, rDash):
        BxDash = 0
        ByDash = 0
        BzDash = 0
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    Q = (-1) ** (i + j + k)
                    xiDash = rDash[0] - self.mxDash[i]
                    yjDash = rDash[1] - self.myDash[j]
                    zkDash = rDash[2] - self.mzDash[k]
                    BxDash += Q * np.arcsinh(
                        zkDash / np.sqrt(np.power(xiDash, 2) + np.power(yjDash, 2))
                    )
                    BzDash += Q * np.arcsinh(
                        (rDash[0] - self.mxDash[k])
                        / np.sqrt(
                            np.power(rDash[2] - self.mzDash[i], 2) + np.power(yjDash, 2)
                        )
                    )
                    ByDash += Q * np.arctan(
                        xiDash
                        * zkDash
                        / (
                            yjDash
                            * np.sqrt(
                                np.power(xiDash, 2)
                                + np.power(yjDash, 2)
                                + np.power(zkDash, 2)
                            )
                        )
                    )
        BDash = (mu0 * self.strength / (4 * pi)) * np.array([-BxDash, ByDash, -BzDash])
        return BDash


class CoilGroup(MagnetGroup):
    """A group of coils
    
    Currently only supports electromagnetic coils, either Circular or Rectangular.
    Stacks coils axially and radially around the same axis. Given a positive value
    of ``axial spacing``, then the coil will grow axially along the positive
    yDash direction
    
    Supports two simulation option: multiple coils within the group, located at
    different physical locations, can be simulated independently. 
    
    Alternatively, the coil can be simulated as a single coil, with
    increased current flowing through it. This is much faster, but less accurate,
    especially where the coil dimensions are such that the spatial extent of the
    individual coils is large relative to the distance to the measurement point
    
    Parameters
    ----------
    strength: float
        Current flowing through coils
    rDash: 3 entry list or array
        Origin of coils in Dash co-ordinate frame
    dimsDash: dict
        Dictionary of coil pair parameters.
        
        * ``spatially distributed``: bool
            Should the programme calculate for each loop independently, or approximate by placing all coils in the same place?
        * ``axial layers``: int
            number of new layers further out along the yDash axis
        * ``axial spacing``: float
            distance between centres of layers in the axial direction
        * ``radial layers``: int
            number of new layers further out away from yDash axis
        * ``radial spacing``: float
            distance between centres of layers in the radial direction
        * ``shape``: str
            Valid entries are variations on ``circ``, ``circular``, ``rectangular``, ``r``, etc
        * Required parameters for either ``RectangularCoil`` or ``CircularCoil``
          (see the documentaiton for those classes)
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self._make_magnets()
        return

    def _make_magnets(self):
        """Create the set of basic magnets specified
        """
        # Shape
        shape = self.dimsDash["shape"].lower()
        if shape in MagnetTypeString.RECT:
            self.base = MagnetType.RECT
        elif shape in MagnetTypeString.CIRC:
            self.base = MagnetType.CIRC
        elif shape in MagnetTypeString.PERM:
            self.base = MagnetType.PERM
            raise NotImplementedError(
                "A group of permanent magnets is not currently supported"
            )
        else:
            raise ValueError(f"Value `{shape}` not understood for keyword `shape`")
        # Verify that other keywords are present
        spatial = self.dimsDash["spatially distributed"]
        layers_ax = self.dimsDash["axial layers"]
        layers_rad = self.dimsDash["radial layers"]
        if spatial:
            spacing_ax = self.dimsDash["axial spacing"]
            spacing_rad = self.dimsDash["radial spacing"]
        # Initialise the relevant magnets
        if spatial:
            # Create many indpendent coils
            for ax in range(layers_ax):
                for rad in range(layers_rad):
                    origin = self.rDash.copy() + (0, ax * spacing_ax, 0)
                    dims = self.dimsDash.copy()
                    if self.base == MagnetType.RECT:
                        # Recall: full length, so expands by double the spacing
                        dims["axDash"] += 2 * rad * spacing_rad
                        dims["azDash"] += 2 * rad * spacing_rad
                    elif self.base == MagnetType.CIRC:
                        dims["radius"] += rad * spacing_rad
                    name = f"{self.name}:{ax}:{rad}"
                    m = self.base(
                        self.strength, origin, dims, self.theta_deg, self.phi_deg, name
                    )
                    self.magnets.append(m)
        else:
            # Create a single coil with an increased current
            name = f"{self.name}:combined"
            m = self.base(
                self.strength * layers_ax * layers_rad,
                self.rDash.copy(),
                self.dimsDash.copy(),
                self.theta_deg,
                self.phi_deg,
                name,
            )
            self.magnets.append(m)
        self.set_colour(self.colour)
        return


class CoilPair(MagnetGroup):
    """A generic pair of CoilGroups, separated in space
    
    Includes space between the two CoilGroups. Supports either Helmholtz or
    anti-Helmholtz current configuration.
    
    Parameters
    ----------
    strength: float
        Current flowing through coils
    rDash: 3 entry list or array
        Origin of coils in Dash co-ordinate frame
    dimsDash: dict
        Dictionary of coil pair parameters.
        
        * ``full spacing``: float
            full spacing between closest coils along yDash axis
        * ``spatially distributed``: bool
            Should the programme calculate for each loop independently, or approximate by placing all coils in the same place?
        * ``axial layers``: int
            number of new layers further out along the yDash axis
        * ``axial spacing``: float
            distance between centres of layers in the axial direction
        * ``radial layers``: int
            number of new layers further out away from yDash axis
        * ``radial spacing``: float
            distance between centres of layers in the radial direction
        * ``configuration``: str
            Valid entries are ``ahh``, ``hh`` and some variations
        * ``shape``: str
            Valid entries are variations on ``circ``, ``circular``, ``rectangular``, ``r``, etc
        * Required parameters for either ``RectangularCoil`` or ``CircularCoil``
          (see the documentaiton for those classes)
    theta: float, optional
        Rotation of Dash frame around Z axis
    phi: float, optional
        Rotation of Dash frame around X axis
    name: str, optional
        Human readable label for the magnet
    """

    def __init__(self, strength, rDash, dimsDash, theta=0, phi=0, name=None):
        super().__init__(strength, rDash, dimsDash, theta, phi, name)
        self._create_magnets()
        return

    def _create_magnets(self):
        """
        Make the actual magnets
        """
        conf_str = self.dimsDash["configuration"].lower()
        if conf_str in PairConfigString.HELMHOLTZ:
            conf = 1
        elif conf_str in PairConfigString.ANTIHELMHOLTZ:
            conf = -1
        else:
            raise ValueError(
                f"Value `{conf_str}` not understood for keyword 'configuration'"
            )

        half_spacing = self.dimsDash["full spacing"] / 2
        spatial = self.dimsDash["spatially distributed"]
        upper_origin = self.rDash.copy() + (0, half_spacing, 0)
        upper_magnet = CoilGroup(
            self.strength,
            upper_origin,
            self.dimsDash,
            self.theta_deg,
            self.phi_deg,
            self.name,
        )
        self.magnets += upper_magnet

        lower_origin = self.rDash.copy() - (0, half_spacing, 0)
        lower_dims = self.dimsDash.copy()
        if spatial:
            lower_dims["axial spacing"] *= -1
        lower_magnet = CoilGroup(
            conf * self.strength,
            lower_origin,
            lower_dims,
            self.theta_deg,
            self.phi_deg,
            self.name,
        )
        self.magnets += lower_magnet
        self.set_colour(self.colour)
        return


class MagnetType:
    RECT = RectangularCoil
    CIRC = CircularCoil
    PERM = PermanentMagnet
    PAIR = CoilPair


class MagnetTypeString:
    RECT = (
        "rect",
        "rectangular",
        "r",
        "square",
        "cube",
        "cubic",
    )
    CIRC = (
        "circ",
        "circular",
        "round",
        "c",
        "spherical",
        "cylindrical",
        "cylinder",
    )
    PERM = (
        "perm",
        "permanent",
        "p",
        "rare-earth",
    )


class PairConfigString:
    HELMHOLTZ = (
        "hh",
        "helmholtz",
        "helm-holtz",
        "homogenous",
    )
    ANTIHELMHOLTZ = (
        "ahh",
        "antihelmholtz",
        "anti-helmholtz",
        "antihelm-holtz",
        "anti-helm-holtz",
    )
