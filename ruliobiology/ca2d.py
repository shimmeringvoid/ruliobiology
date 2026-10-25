"""
ruliobiology.ca2d -- Wolfram's ruliobiology model in two dimensions.

A 2-D rulioorganism is a k-colour (default 4) **outer-totalistic** cellular
automaton: a cell's next colour depends on its own colour and on how many
of its neighbours have each colour, on either the von Neumann (4-cell,
the default) or the Moore (8-cell) neighbourhood.  Such rules are
automatically symmetric under every rotation and reflection of the square
(the dihedral group D4) -- the 2-D analogue of the left-right symmetry of
the 1-D model -- so an organism grown from a single seed cell stays
D4-symmetric for its whole life.  Its "form" is the (t, y, x) space-time
volume; see viewer3d.py to rotate it and look inside.

Genome size (independent rule cases):
    von Neumann:  k=3: 45    k=4: 140   k=5: 350
    Moore:        k=3: 135   k=4: 660   k=5: 2475   (k=2: 18, the Life-like rules)
Bigger genomes evolve much more slowly, because a mutation only has an
effect when it hits one of the few cases a small organism actually uses
(1-D: 75 cases, ~half of them used).  k=4 von Neumann or k=3 Moore get
going in a few thousand steps; k=4 Moore needs tens of thousands.

Infinite patterns: periodic ones are caught exactly; a pattern that grows
wider than ``max_width`` (default 100) or outlives ``max_steps`` (default
1000) counts as infinite.  In 2-D the width cap is the one that matters
for speed, since a pattern growing at light speed costs ~t**3 cell updates.

Everything else -- lifetime fitness, point mutations, the
accept-if-not-worse loop, the Evolution record -- is shared with the 1-D
module (``ruliobiology.ca1d``).

Quick start
-----------
>>> import ruliobiology as rb                        # the package re-exports this module
>>> ev = rb.evolve2d(steps=10000, seed=1)            # ~10-30 s with numba
>>> vol = rb.grow2d(ev.rule)                         # (lifetime, H, W) uint8 volume
>>> rb.plot_filmstrip(vol); rb.plot_projection(vol)
>>> rb.write_viewer(vol, "organism.html")            # rotatable 3-D view
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from . import ca1d as rb
from .ca1d import _njit, HAVE_NUMBA, PALETTE, colormap, Evolution, fit_lifetime

DEFAULT_K = 4
DEFAULT_NEIGHBORHOOD = "vonneumann"
DEFAULT_SEED_COLOR = 1
DEFAULT_MAX_STEPS = 1000   # generations after which a pattern counts as infinite
DEFAULT_MAX_WIDTH = 100    # bounding-box side beyond which a pattern counts as infinite

# ----------------------------------------------------------------------------
# rule-case bookkeeping
# ----------------------------------------------------------------------------

def _M(k: int) -> int:
    """Stride of the 'own colour' digit in the lookup table (9**(k-1))."""
    return 9 ** (k - 1)


NEIGHBORHOODS = {
    "moore": ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
    "vonneumann": ((-1, 0), (0, -1), (0, 1), (1, 0)),
}


@lru_cache(maxsize=None)
def valid_cases(k: int = DEFAULT_K, n_neighbors: int = 8):
    """All independent cases (own, counts) of an outer-totalistic rule:
    own colour, and a tuple (n_1, ..., n_{k-1}) of neighbour counts for the
    non-white colours with n_1 + ... + n_{k-1} <= n_neighbors (the rest are
    white).  Fixed order; (0, (0,...,0)) -- the all-white neighbourhood --
    is first."""
    cases = []
    for own in range(k):
        for counts in itertools.product(range(n_neighbors + 1), repeat=k - 1):
            if sum(counts) <= n_neighbors:
                cases.append((own, counts))
    return tuple(cases)


def case_index(own: int, counts: Sequence[int], k: int = DEFAULT_K) -> int:
    """Index into the full lookup table of case (own, counts)."""
    i = own * _M(k)
    for c, n in enumerate(counts):
        i += n * 9 ** c
    return i


def index_case(i: int, k: int = DEFAULT_K):
    """Inverse of :func:`case_index`."""
    own, rest = divmod(i, _M(k))
    counts = []
    for _ in range(k - 1):
        counts.append(rest % 9)
        rest //= 9
    return own, tuple(counts)


class Rule2D:
    """A k-colour outer-totalistic Moore-neighbourhood rule.

    ``table`` has k * 9**(k-1) entries indexed by :func:`case_index`; only the
    entries of :func:`valid_cases` are ever read.  The all-white case is
    pinned to white so that finite patterns on an infinite background make
    sense.
    """

    __slots__ = ("table", "k", "neighborhood")

    def __init__(self, table, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD):
        table = np.ascontiguousarray(table, dtype=np.uint8)
        if len(table) != k * _M(k):
            raise ValueError(f"table has {len(table)} entries, expected {k * _M(k)}")
        if table.max(initial=0) >= k:
            raise ValueError("table contains colours >= k")
        if neighborhood not in NEIGHBORHOODS:
            raise ValueError(f"neighborhood must be one of {list(NEIGHBORHOODS)}")
        self.table = table
        self.k = k
        self.neighborhood = neighborhood

    @property
    def offsets(self):
        return NEIGHBORHOODS[self.neighborhood]

    @property
    def n_neighbors(self) -> int:
        return len(self.offsets)

    @property
    def valid_cases(self):
        return valid_cases(self.k, self.n_neighbors)

    @property
    def n_cases(self) -> int:
        return len(self.valid_cases)

    # -- constructors --------------------------------------------------------
    @classmethod
    def null(cls, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD) -> "Rule2D":
        return cls(np.zeros(k * _M(k), dtype=np.uint8), k, neighborhood)

    @classmethod
    def from_cases(cls, cases, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD) -> "Rule2D":
        """Build from a mapping {(own, counts): colour}; unspecified cases
        are white.  ``counts`` lists the neighbour counts of colours 1..k-1."""
        r = cls.null(k, neighborhood)
        for (own, counts), out in dict(cases).items():
            if sum(counts) > r.n_neighbors or len(counts) != k - 1:
                raise ValueError(f"bad case {(own, counts)}")
            r.table[case_index(own, counts, k)] = out
        return r

    @classmethod
    def life_like(cls, birth=(3,), survive=(2, 3)) -> "Rule2D":
        """k = 2: Conway-style B/S rule (default B3/S23 = the Game of Life)."""
        cases = {}
        for n in range(9):
            cases[(0, (n,))] = 1 if n in birth else 0
            cases[(1, (n,))] = 1 if n in survive else 0
        return cls.from_cases(cases, k=2, neighborhood="moore")

    @classmethod
    def random(cls, rng=None, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD,
               fix_background: bool = True) -> "Rule2D":
        rng = np.random.default_rng(rng)
        cases = valid_cases(k, len(NEIGHBORHOODS[neighborhood]))
        outs = rng.integers(0, k, size=len(cases))
        if fix_background:
            outs[0] = 0
        return cls.from_cases(dict(zip(cases, outs)), k, neighborhood)

    @classmethod
    def from_number(cls, number: int, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD) -> "Rule2D":
        cases = valid_cases(k, len(NEIGHBORHOODS[neighborhood]))
        digits = []
        for _ in cases:
            digits.append(number % k)
            number //= k
        if number:
            raise ValueError("rule number too large for this k")
        return cls.from_cases(dict(zip(cases, reversed(digits))), k, neighborhood)

    # -- properties ----------------------------------------------------------
    def to_number(self) -> int:
        """The outputs of all independent cases (in :func:`valid_cases` order,
        first case most significant) read as a base-k integer."""
        n = 0
        for own, counts in self.valid_cases:
            n = n * self.k + int(self.table[case_index(own, counts, self.k)])
        return n

    def cases(self):
        return [((own, counts), int(self.table[case_index(own, counts, self.k)]))
                for own, counts in self.valid_cases]

    def __call__(self, own: int, counts: Sequence[int]) -> int:
        return int(self.table[case_index(own, counts, self.k)])

    def copy(self) -> "Rule2D":
        return Rule2D(self.table.copy(), self.k, self.neighborhood)

    def __eq__(self, other):
        return (isinstance(other, Rule2D) and self.k == other.k
                and self.neighborhood == other.neighborhood
                and np.array_equal(self.table, other.table))

    def __hash__(self):
        return hash((self.k, self.neighborhood, self.table.tobytes()))

    def __repr__(self):
        return f"Rule2D(k={self.k}, neighborhood={self.neighborhood!r}, number={self.to_number()})"

    def mutated(self, rng=None, n_mutations: int = 1, fix_background: bool = True) -> "Rule2D":
        """Copy with ``n_mutations`` distinct cases changed to a different
        random colour (the all-white case is left alone with fix_background)."""
        rng = np.random.default_rng(rng)
        cases = self.valid_cases
        first = 1 if fix_background else 0
        picks = rng.choice(len(cases) - first, size=n_mutations, replace=False) + first
        table = self.table.copy()
        for p in picks:
            own, counts = cases[p]
            i = case_index(own, counts, self.k)
            table[i] = (int(table[i]) + int(rng.integers(1, self.k))) % self.k
        return Rule2D(table, self.k, self.neighborhood)

    def distance(self, other: "Rule2D") -> int:
        return sum(1 for (_, x), (_, y) in zip(self.cases(), other.cases()) if x != y)


# ----------------------------------------------------------------------------
# phenotype statistics
# ----------------------------------------------------------------------------

class Stats2D(NamedTuple):
    """lifetime (0 = never dies), width (largest bounding-box side),
    max_area (most cells alive in one generation), counts per colour over
    the whole history (counts[0] = white cells inside the bounding boxes)."""
    lifetime: int
    width: int
    max_area: int
    counts: tuple

    @property
    def finite(self) -> bool:
        return self.lifetime > 0

    @property
    def cells(self) -> int:
        return int(sum(self.counts[1:]))

    @property
    def aspect(self) -> float:
        return self.width / self.lifetime if self.lifetime else 0.0

    @property
    def color_fractions(self) -> np.ndarray:
        f = np.array(self.counts, dtype=float)
        f[0] = 0.0
        s = f.sum()
        return f / s if s else f


@_njit(cache=True, nogil=True)
def _stats2d_kernel(table, k, M, pow9, dxs, dys, seed, max_steps, max_width):
    """Simulate one quadrant (x >= 0, y >= 0) of the D4-symmetric pattern.
    Returns (lifetime, width, max_area, counts); lifetime 0 = infinite."""
    R = max_steps + 4
    cur = np.zeros((R, R), np.uint8)
    nxt = np.zeros((R, R), np.uint8)
    cur[0, 0] = seed
    hi = 0
    counts = np.zeros(k, np.int64)
    counts[seed] += 1
    n = np.zeros(k, np.int64)
    widest = 1
    max_area = 1
    seen = {np.int64(0)}
    for t in range(1, max_steps + 1):
        h = np.int64(-3750763034362895579)
        for x in range(hi + 1):
            for y in range(hi + 1):
                h = (h ^ np.int64(cur[x, y])) * np.int64(1099511628211)
        h ^= np.int64(hi)
        if h in seen:
            return 0, widest, max_area, counts
        seen.add(h)
        nhi = -1
        area = 0
        for x in range(hi + 2):
            for y in range(hi + 2):
                for c in range(k):
                    n[c] = 0
                for j in range(dxs.shape[0]):
                    xx = x + dxs[j]
                    if xx < 0:
                        xx = -xx
                    yy = y + dys[j]
                    if yy < 0:
                        yy = -yy
                    n[cur[xx, yy]] += 1
                idx = cur[x, y] * M
                for c in range(1, k):
                    idx += n[c] * pow9[c - 1]
                v = table[idx]
                nxt[x, y] = v
                if v != 0:
                    if x > nhi:
                        nhi = x
                    if y > nhi:
                        nhi = y
                    mult = 4
                    if x == 0:
                        mult //= 2
                    if y == 0:
                        mult //= 2
                    counts[v] += mult
                    area += mult
        for x in range(hi + 1):
            for y in range(hi + 1):
                cur[x, y] = 0
        if nhi < 0:
            return t, widest, max_area, counts
        for x in range(nhi + 1):                 # white cells inside the box
            for y in range(nhi + 1):
                if nxt[x, y] == 0:
                    mult = 4
                    if x == 0:
                        mult //= 2
                    if y == 0:
                        mult //= 2
                    counts[0] += mult
        if area > max_area:
            max_area = area
        w = 2 * nhi + 1
        if w > widest:
            widest = w
        if w > max_width:
            return 0, widest, max_area, counts
        cur, nxt = nxt, cur
        hi = nhi
    return 0, widest, max_area, counts


def _step2d(grid: np.ndarray, table: np.ndarray, k: int, offsets=None) -> np.ndarray:
    """One generation on the whole plane.  ``grid`` is the bounding-box crop;
    the result covers the box enlarged by one cell on every side."""
    M = _M(k)
    offsets = NEIGHBORHOODS[DEFAULT_NEIGHBORHOOD] if offsets is None else offsets
    P = np.pad(grid, 2)
    H, W = P.shape
    idx = P[1:-1, 1:-1].astype(np.int64) * M
    for c in range(1, k):
        m = (P == c).astype(np.int64)
        n = np.zeros((H - 2, W - 2), dtype=np.int64)
        for dy, dx in offsets:
            n += m[1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
        idx += n * 9 ** (c - 1)
    return table[idx]


def _trim(grid: np.ndarray):
    """Crop to the bounding box of the non-white cells -> (rows, cols, crop)
    or None if the grid is all white."""
    rows = np.flatnonzero(grid.any(axis=1))
    cols = np.flatnonzero(grid.any(axis=0))
    if rows.size == 0:
        return None
    return rows[0], cols[0], grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _stats2d_numpy(table, k, offsets, seed, max_steps, max_width):
    grid = np.array([[seed]], dtype=np.uint8)
    counts = np.zeros(k, dtype=np.int64)
    counts[seed] += 1
    widest = max_area = 1
    seen = set()
    for t in range(1, max_steps + 1):
        key = (grid.shape, grid.tobytes())
        if key in seen:
            return 0, widest, max_area, counts
        seen.add(key)
        tr = _trim(_step2d(grid, table, k, offsets))
        if tr is None:
            return t, widest, max_area, counts
        grid = tr[2]
        area = int(np.count_nonzero(grid))
        max_area = max(max_area, area)
        widest = max(widest, *grid.shape)
        if max(grid.shape) > max_width:
            return 0, widest, max_area, counts
        counts += np.bincount(grid.ravel(), minlength=k)
    return 0, widest, max_area, counts


def stats2d(rule: Rule2D, max_steps: int = DEFAULT_MAX_STEPS, max_width: int = DEFAULT_MAX_WIDTH,
            seed_color: int = DEFAULT_SEED_COLOR, engine: str = "auto") -> Stats2D:
    """Grow the rule from a single seed and summarise the result."""
    if engine == "auto":
        engine = "numba" if HAVE_NUMBA else "numpy"
    if engine == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed")
        k = rule.k
        pow9 = np.array([9 ** c for c in range(k - 1)], dtype=np.int64)
        dys = np.array([o[0] for o in rule.offsets], dtype=np.int64)
        dxs = np.array([o[1] for o in rule.offsets], dtype=np.int64)
        out = _stats2d_kernel(rule.table, k, _M(k), pow9, dxs, dys, seed_color,
                              max_steps, max_width)
    elif engine == "numpy":
        out = _stats2d_numpy(rule.table, rule.k, rule.offsets, seed_color, max_steps, max_width)
    else:
        raise ValueError("engine must be 'auto', 'numba' or 'numpy'")
    life, width, area, counts = out
    return Stats2D(int(life), int(width), int(area), tuple(int(c) for c in counts))


def lifetime2d(rule: Rule2D, **kwargs) -> int:
    return stats2d(rule, **kwargs).lifetime


def grow2d(rule: Rule2D, steps: Optional[int] = None, initial=None,
           seed_color: int = DEFAULT_SEED_COLOR, max_steps: int = DEFAULT_MAX_STEPS,
           max_width: int = DEFAULT_MAX_WIDTH) -> np.ndarray:
    """Space-time history as a (generations, H, W) uint8 volume, every
    generation placed in a common frame (the bounding box of the whole life).

    steps   : generations to compute after the initial one (None = until the
              pattern dies, at most max_steps).
    initial : optional 2-D array to start from instead of a single seed.
    """
    table, k = rule.table, rule.k
    if initial is None:
        grid = np.array([[seed_color]], dtype=np.uint8)
        top = left = 0
    else:
        tr = _trim(np.asarray(initial, dtype=np.uint8))
        if tr is None:
            return np.zeros((1, 1, 1), dtype=np.uint8)
        top, left, grid = tr
    frames = [(top, left, grid)]
    for _ in range(max_steps if steps is None else steps):
        new = _step2d(grid, table, k, rule.offsets)
        tr = _trim(new)
        if tr is None:
            break
        top, left = top - 1 + tr[0], left - 1 + tr[1]
        grid = tr[2]
        frames.append((top, left, grid))
        if max(grid.shape) > max_width:
            break
    t0 = min(f[0] for f in frames)
    l0 = min(f[1] for f in frames)
    H = max(f[0] + f[2].shape[0] for f in frames) - t0
    W = max(f[1] + f[2].shape[1] for f in frames) - l0
    vol = np.zeros((len(frames), H, W), dtype=np.uint8)
    for t, (r, c, g) in enumerate(frames):
        vol[t, r - t0:r - t0 + g.shape[0], c - l0:c - l0 + g.shape[1]] = g
    return vol


# ----------------------------------------------------------------------------
# fitness functions specific to 2-D (the 1-D ones -- fit_lifetime, fit_width,
# fit_tall, fit_wide, fit_cells, fit_color -- work unchanged on Stats2D)
# ----------------------------------------------------------------------------

def fit_area(s: Stats2D) -> float:
    """Be as big as possible at some moment (then die)."""
    return s.max_area if s.finite else 0


def fit_volume(s: Stats2D) -> float:
    """Occupy as much space-time as possible."""
    return s.cells if s.finite else 0


# ----------------------------------------------------------------------------
# adaptive evolution (the 1-D loop, with a 2-D evaluator)
# ----------------------------------------------------------------------------

def evolve2d(steps: int = 10000, fitness: Callable = fit_lifetime, start: Optional[Rule2D] = None,
             seed=None, n_mutations: int = 1, k: int = DEFAULT_K, neighborhood: str = DEFAULT_NEIGHBORHOOD,
             max_steps: int = DEFAULT_MAX_STEPS, max_width: int = DEFAULT_MAX_WIDTH,
             seed_color: int = DEFAULT_SEED_COLOR, fix_background: bool = True,
             progress: bool = False, callback: Optional[Callable] = None) -> Evolution:
    """Adaptive evolution of a 2-D rulioorganism; see ``ruliobiology.ca1d.evolve``."""
    start = start if start is not None else Rule2D.null(k, neighborhood)

    def evaluate(rule):
        return stats2d(rule, max_steps=max_steps, max_width=max_width, seed_color=seed_color)

    ev = rb.evolve(steps=steps, fitness=fitness, start=start, seed=seed,
                   n_mutations=n_mutations, fix_background=fix_background,
                   progress=progress, evaluate=evaluate, callback=callback)
    ev.params.update(k=start.k, neighborhood=start.neighborhood, max_steps=max_steps,
                     max_width=max_width, seed_color=seed_color, dimension=2)
    return ev


def _job(args):
    seed, kwargs = args
    return evolve2d(seed=seed, **kwargs)


def evolve2d_many(n_runs: int, steps: int = 10000, seed=0, n_jobs: int = 1, **kwargs):
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    jobs = [(s, dict(steps=steps, **kwargs)) for s in seeds]
    if n_jobs == 1:
        return [_job(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_job, jobs))


# ----------------------------------------------------------------------------
# 2-D pictures (for the rotatable 3-D view see viewer3d.py)
# ----------------------------------------------------------------------------

def projection(vol: np.ndarray, mode: str = "last_color") -> np.ndarray:
    """Collapse the (t, y, x) volume to a picture.
    'last_color' : colour each cell had when it was last alive (0 = never);
    'birth'      : generation at which each cell first became non-white (+1);
    'death'      : generation at which each cell was last non-white (+1);
    'count'      : how many generations each cell was non-white."""
    alive = vol > 0
    if mode == "count":
        return alive.sum(axis=0)
    T = vol.shape[0]
    tt = np.arange(1, T + 1)[:, None, None]
    if mode == "birth":
        return np.where(alive.any(0), np.where(alive, tt, T + 1).min(0), 0)
    last = np.where(alive, tt, 0).max(0)                  # 0 = never alive
    if mode == "death":
        return last
    if mode == "last_color":
        out = np.zeros(vol.shape[1:], dtype=np.uint8)
        idx = np.nonzero(last)
        out[idx] = vol[last[idx] - 1, idx[0], idx[1]]
        return out
    raise ValueError(mode)


def plot_projection(vol: np.ndarray, mode: str = "last_color", ax=None, k: int = DEFAULT_K,
                    title: Optional[str] = None, max_inches: float = 5.0, min_size: int = 12):
    """Draw a projection of the life of a 2-D organism.  Tiny organisms are
    padded to ``min_size`` cells so that they stay tiny in a gallery."""
    import matplotlib.pyplot as plt
    img = projection(vol, mode)
    if img.shape[0] < min_size or img.shape[1] < min_size:
        ph, pw = max(min_size - img.shape[0], 0), max(min_size - img.shape[1], 0)
        img = np.pad(img, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
    if ax is None:
        _, ax = plt.subplots(figsize=(max_inches, max_inches))
    if mode == "last_color":
        ax.imshow(img, cmap=colormap(k), vmin=0, vmax=k - 1, interpolation="nearest")
    else:
        m = np.ma.masked_equal(img, 0)
        ax.imshow(m, cmap="viridis", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or {"last_color": "colour when last alive", "birth": "birth time",
                           "death": "death time", "count": "generations alive"}[mode],
                 fontsize=9)
    return ax


def plot_filmstrip(vol: np.ndarray, n: int = 12, k: int = DEFAULT_K, cell: float = 1.6):
    """``n`` evenly spaced generations of a 2-D organism, side by side."""
    import matplotlib.pyplot as plt
    T = vol.shape[0]
    ts = np.unique(np.linspace(0, T - 1, min(n, T)).round().astype(int))
    fig, axes = plt.subplots(1, len(ts), figsize=(cell * len(ts), cell * 1.2), squeeze=False)
    for ax, t in zip(axes[0], ts):
        ax.imshow(vol[t], cmap=colormap(k), vmin=0, vmax=k - 1, interpolation="nearest")
        ax.set_title(f"t = {t}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return fig


def plot_gallery2d(rules: Sequence[Rule2D], ncols: int = 6, titles: Optional[Sequence] = None,
                   mode: str = "last_color", max_steps: int = DEFAULT_MAX_STEPS,
                   max_width: int = DEFAULT_MAX_WIDTH, cell: float = 2.0):
    """Grid of projections of many organisms."""
    import matplotlib.pyplot as plt
    n = len(rules)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            vol = grow2d(rules[i], max_steps=max_steps, max_width=max_width)
            plot_projection(vol, mode, ax=ax, k=rules[i].k,
                            title="" if titles is None else titles[i])
            ax.axis("off")
    fig.tight_layout()
    return fig


__all__ = ["DEFAULT_K", "DEFAULT_NEIGHBORHOOD", "DEFAULT_MAX_STEPS", "DEFAULT_MAX_WIDTH", "DEFAULT_SEED_COLOR",
           "NEIGHBORHOODS", "Rule2D", "Stats2D",
           "valid_cases", "case_index", "index_case", "stats2d", "lifetime2d", "grow2d",
           "fit_area", "fit_volume", "evolve2d", "evolve2d_many",
           "projection", "plot_projection", "plot_filmstrip", "plot_gallery2d"]
