# mfs

[![Build Status](https://travis-ci.com/simon-ball/nqo-mfs.svg?branch=master&status=passed)](https://travis-ci.com/simon-ball/nqo-mfs)
[![codecov](https://codecov.io/gh/simon-ball/nqo-mfs/branch/master/graph/badge.svg)](https://codecov.io/gh/simon-ball/nqo-mfs)
[![Documentation Status](https://readthedocs.org/projects/nqo-mfs/badge/?version=latest)](https://nqo-mfs.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


`mfs` is a library for calculating and visualising magnetic fields for neutral atom trapping. (C) Simon Ball 2018-2020

Originally written to assist in designing the Ytterbium quantum optics experiment at the University of Southern Denmark (SDU) in 2018. Cleaned up and published in 2020 for use by a project at the Norwegian University of Science and Technology (NTNU)

Install directly from `github` with pip:

    pip install git+https://github.com/simon-ball/nqo-mfs


# Usage

## Sources
`mfs` implements four types of magnetic field source:

- `mfs.sources.PermanentMagnet` : A cuboid permanent magnet.

- `mfs.sources.CircularCoil` : A single circular loop of wire carrying a current

- `mfs.sources.RectangularCoil` : A single rectangualr loop of wire carrying a current

- `mfs.sources.CoilPair` : A pair of coils, each containing 1 or more individual coils, either rectangular or circular

## Geometry
Each magnetic field source is generated in their own frame of reference (referred to by dashes throughout), controlled by two angles, `theta` and `phi`. All angles are handled in degrees. 

- `theta` : azimuthal angle
  If `phi` is 0°, then `theta` corresponds to a rotation of the X'-Y' plane around the Z = Z' axis
  
- `phi` : polar angle
  If `theta` is 0°, then `phi` corresponds to a rotation of the Y'-Z' plane around the X = X' axis

Rectangular magnet field sources are always square to the X'-Y'-Z' axes

Rotation calculations can be adjusted to cope with a rotation around the Y = Y' axis if needed, but are not currently implemented

Rotations are implemented around the (0,0,0) origin. Rotations around arbitrary origins are not currently supported. 


## Magnetic axis
`mfs` uses the convention that the magnetisation points along the +ve Y' axis: 

- permanent magnets have their magnetisation pointing along the Y' axis

- Circular and rectangular magnets are in the X'-Z' plane, wrapped around the Y' axis

- A negative value for strength reverses the direction

## Units
SI units are used throught - distances are in metres, current is in Teslas, current in Amps, and magnetisation in Tesla/metres

Angles are always in degrees. 

# References
## Permanent magnets

https://link.springer.com/content/pdf/10.1007%2FBF01573988.pdf
This function implements the approach taken by Metzger in Archiv fur Elektrotechnik 59 (1977) 229-242.

## Circular coils

https://doi.org/10.1103/PhysRevA.35.1535
Bergeman, Magnetostatic trapping fields for neutral atoms, 1987, PRA 35 1535

## Rectangular coils

https://dx.doi.org/10.6028%2Fjres.105.045
Misiaken, Equations for the Magnetic Field produced by One or More Rectangualr Loops of Wire in the Same Plane
J Res Natl Inst Stand Technol. 2000 Jul-Aug; 105(4): 557–564