"""Optional external EDA tools (ngspice, Xschem, KiCad, KLayout, OpenFOAM) behind one interface.

ChipLab never reimplements these tools. It writes their input files from fixed templates or
from the studio's design record, runs the real binaries with fixed argument lists (no shell,
no client-supplied commands), and parses their output. Every tool is optional:
:func:`chiplab.tools.registry.status` says which are present and the studio hides the rest.
"""
