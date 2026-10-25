"""
ruliobiology -- Stephen Wolfram's ruliobiology model of adaptive evolution
(simple cellular-automaton rules evolved by mutation and selection), in one
and two dimensions, with a rotatable 3-D viewer for the 2-D organisms.

    import ruliobiology as rb

    ev = rb.evolve(steps=1000, seed=1)              # 1-D: Wolfram's setup
    rb.plot_breakthroughs(ev); rb.plot_organism(ev.rule)

    ev2 = rb.evolve2d(steps=15000, seed=3)          # 2-D: k=4, von Neumann
    vol = rb.grow2d(ev2.rule)                       # (t, y, x) space-time volume
    rb.write_viewer(vol, "organism.html")           # rotate it in a browser

The submodules hold the details:  ``ruliobiology.ca1d``, ``ruliobiology.ca2d``,
``ruliobiology.viewer3d``.  Everything below is re-exported for convenience.

Reference:  S. Wolfram, "'Endless Forms Most Beautiful and Most Wonderful':
Ruliobiology and the Theoretical Foundations of Biological Form" (Oct 2026),
https://writings.stephenwolfram.com/2026/10/endless-forms-most-beautiful-and-most-wonderful-ruliobiology-and-the-theoretical-foundations-of-biological-form/
"""
from .ca1d import (  # noqa: F401
    DEFAULT_K, DEFAULT_MAX_STEPS, DEFAULT_SEED_COLOR, HAVE_NUMBA,
    Rule, Stats, Evolution, symmetric_cases, case_index, index_case,
    stats, lifetime, grow, used_cases,
    fit_lifetime, fit_width, fit_wide, fit_tall, fit_cells, fit_color,
    evolve, evolve_many,
    PALETTE, colormap, plot_organism, plot_fitness, plot_gallery,
    plot_breakthroughs, plot_development, plot_rule,
)
from .ca2d import (  # noqa: F401
    NEIGHBORHOODS, Rule2D, Stats2D, valid_cases,
    stats2d, lifetime2d, grow2d, fit_area, fit_volume, evolve2d, evolve2d_many,
    projection, plot_projection, plot_filmstrip, plot_gallery2d,
)
from .viewer3d import write_viewer, rle_encode  # noqa: F401

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
