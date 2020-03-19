# mfs

`mfs` is a library for calculating and visualising magnetic fields for neutral atom trapping. (C) Simon Ball 2018-2020

Originally written to assist in designing the Ytterbium quantum optics experiment at the University of Southern Denmark (SDU) in 2018. Cleaned up and published in 2020 for use by a project at the Norwegian University of Science and Technology (NTNU)


# Usage

## Sources
`mfs` implements four types of magnetic field source:
    * `mfs.sources.PermanentMagnet` : A cuboid permanent magnet. 
    * `mfs.sources.CircularCoil` : A single circular loop of wire carrying a current
    * `mfs.sources.RectangularCoil` : A single rectangualr loop of wire carrying a current
    * `mfs.sources.CoilPair` : A pair of coils, each containing 1 or more individual coils, either rectangular or circular

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

## Units
SI units are used throught - distances are in metres, current is in Teslas, current in Amps, and magnetisation in Tesla/metres

Angles are always in degrees. 


## Initalising magnets
Each geometry specific magnet class is based on a generic template:
    
    magnet = mfs.sources.MagnetClass(strength, rDash, dimsDash, theta, phi)
    
where:
`strength` : float
    Magnetisation or current
`rDash`
    Origin (centre) of the magnet **in the frame of the magnet**, i.e. rotated by `theta` and `phi` with respect to the laboratory frame
`dimsDash`
    Dictionary of magnet specific parameters
`theta`, `phi` : float  
    Angle of the magnet frame of reference with respect to the laboratory frame of reference

### PermanentMagnet
`strength` is given in Tesla/metre

`dimsDash` requires the following keywords:
    - `"axDash"` : float
        Full length of the magnet in metres in the X' direction
    - `"ayDash"` : float
        Full length of the magnet in metres in the Y' direction
    - `"azDash"` : float
        Full length of the magnet in metres in the Z' direction


### CircularCoil
`strength` is given in Amps. Positive current implies current flowing such that the magnetisation is in the +ve Y' direction. Provide a negative current to reverse the direction

`dimsDash` requires the following keywords:
    - `radius` : float
        Radius of the coil, from origin to centre of conductor

### RectangularCoil
`strength` is given in Amps. Positive current implies current flowing such that the magnetisation is in the +ve Y' direction. Provide a negative current to reverse the direction

`dimsDash` requires the following keywords:
    - `"axDash"` : float
        Full length of the coil in metres in the X' direction, from conductor-centre to conductor-centre
    - `"azDash"` : float
        Full length of the coil in metres in the z' direction, from conductor-centre to conductor-centre

### CoilPair
`strength` is given in Amps. Positive current implies current flowing such that the magnetisation is in the +ve Y' direction, away from the origin. Provide a negative current to reverse the direction

dimsDash requires the following keywords **in addition to** the geometry specific keywords given above:
    - `"shape"` : str
        All coils are circular or rectangular. Accepted values are `circ`, `rect` and some variations Relevant circular or rectangular parameters given above are required in addition to keywords below
    - `"configuration"` : strength
        Helmholtz or anti-helmholtz. Accepted values are `hh`, `ahh`, and some variations. Assuming the `ahh` and that `strength` is positive, then the magnetisiation will flow out from the coil origin along the Y' axis, and return in the X'-Z' plane.
    - `"full spacing"` : float
        Distance in metres along the Y' axis between the origins of the top and bottom coils. If `spatially distributed` is `True`, then this is the **closest distance from conductor-centre to conductor-centre**. 
    - `"axial layers"` : int
        Number of layers of windings along the axial direction, in **each** coil
    - `"radial layers"` : int
        Number of layers of windings along the radial direction, in **each** coil
    - `"spatially distributed"` : bool
        Flag to set the detail of the simulation. If `True`, then each coil of wire is simulated independently, based on the specified `axial spacing` and `radial spacing`. In this case, the additional axial layers exist further away from the origin of the CoilPair.
        Can result in dramatically slower calculations. NB: coils are modelled as independent rings, each carrying the same current, **not** as a spiral. 
        If `False`, then the coil is modelled as a single loop of wire with `strength` increased by a factor of `axial_layers * radial_layers`. Significantly quicker to calculate, but less accurate. 
    - `"axial spacing"` : float
        if `spatially distributed` is true, use this value of offset each additional coil in the axial direction
    - `"radial spacing"` : float
        if `spatially distributed` is true, use this value of offset each additional coil in the radial direction
    

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