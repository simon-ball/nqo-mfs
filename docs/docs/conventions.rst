Conventions and Notation
========================

To maintain consistency, the following conventions are used throughout the library


Frames of reference
*******************

In general, the magnets are implemented in their own, rotated frame of reference (referred to by dashes throughout). When each magnet instance is generated, this frame is defined by two angles, ``theta`` and ``phi``. Both are given in **degrees**.

Rotations are all right-handed, i.e. a positive angle rotation around an axis results in a counter-clockwise movement, when viewed *from* positive-infinity on that axis





-----
Theta
-----

``theta`` is the azimuthal angle, i.e. rotation around the global ``Z`` axis. If ``phi=0°``, then  the ``X'-Y'`` plane is in the same plane as the ``X-Y`` frame, but rotated around the ``Z=Z'`` axis. 


---
Phi
---

``phi`` is the polar angle, i.e. the rotation around the global ``X`` axis. If ``theta=0°``, then the ``Y'-Z'`` plane is in the same plane as the ``Y-Z`` frame, but rotated around the ``X=X'`` axis.

-----
Order
-----

Due to the use of extrinsic angles, the order of operations is important. When a frame is specified with both ``theta != 0``, ``phi != 0``, the frame is first rotated around the ``X`` axis by angle ``phi``, and second rotated around the ``Z`` axis by angle ``theta``

----
Note
----

Rotation around the global ``Y`` axis is not currently supported. 

All rotations are spherical, i.e. around the origin ``(0, 0, 0)`` in the global frame. Consequently, this package does not (currently) support completely arbitrary magnets, as the experiment it was written to support did not require that degree of complexity. 


Principal Axes
**************
By convention, all magnets are defined with respect to the ``Y'`` axis
* Rectangular permanent magnets have their magnetisation pointing along the ``Y'`` axis
* Electromagnetic coils have their windings parallel to the ``X'-Z'`` plane, wrapped around the ``Y'`` axis

Units
******
SI units are used throughout - metres, Tesla, Amps, etc. 

Rectilinear magnets are always specifed by their **full** length.

Circular magnets are always specifed by their **radius**


Handling groups of magnets
**************************

As a general rule, methods which operate on a magnet will accept either a single magnet, or an iterable (list, tuple etc) containing multiple magnets.
