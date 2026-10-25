"""
ruliobiology.ca1d -- Stephen Wolfram's "ruliobiology" model of adaptive
evolution, in one dimension (the setup of his essays).

    S. Wolfram, "'Endless Forms Most Beautiful and Most Wonderful':
    Ruliobiology and the Theoretical Foundations of Biological Form"
    (Oct 2026), building on "Why Does Biological Evolution Work?" (May 2024)
    and "Foundations of Biological Evolution: More Results & More Surprises"
    (Dec 2024).

The model
---------
* A **rulioorganism** is a one-dimensional, k-colour (default k = 5),
  nearest-neighbour cellular automaton.  Colour 0 is the white background.
* The rule is **left-right symmetric**: the output for the neighbourhood
  (a, b, c) equals the output for (c, b, a).  For k = 5 there are
  5 * 5 * 6 / 2 = 75 independent rule cases ("ruliogenome" of length 75).
* The **genotype** is the rule table; the **phenotype** is the space-time
  pattern grown from a single seed cell on an otherwise white infinite line.
* The all-white neighbourhood is pinned to white, so that a finite pattern
  on an infinite background is well defined.
* **Fitness** is, by default, the *lifetime*: the number of generations the
  pattern survives before every cell is white again.  Patterns that never
  die (periodic ones, detected exactly; or ones still alive after
  ``max_steps`` generations / wider than ``max_width``) get fitness 0.
* **Adaptive evolution** is a single asexual lineage: at each step one
  (or a few) randomly chosen rule case(s) is changed to a random different
  colour; the mutant *replaces* the parent if its fitness is >= the parent's
  (so neutral mutations drift), otherwise it is discarded.  There is no
  population, no recombination and no environment -- that is the point.

Requirements:  numpy, matplotlib.  numba is optional but makes the fitness
evaluation ~5-10x faster (``pip install numba``).

Quick start
-----------
>>> import ruliobiology as rb                   # the package re-exports this module
>>> ev = rb.evolve(steps=1000, seed=1)          # ~seconds with numba
>>> ev.fitness                                  # lifetime of the final organism
>>> rb.plot_fitness(ev); rb.plot_breakthroughs(ev); rb.plot_organism(ev.rule)
>>> import matplotlib.pyplot as plt; plt.show()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence

import numpy as np

try:  # optional JIT acceleration
    from numba import njit as _njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def _njit(*args, **kwargs):
        def deco(f):
            return f
        return deco if not (args and callable(args[0])) else args[0]


DEFAULT_K = 5             # number of colours, as in Wolfram's 2026 essay
DEFAULT_SEED_COLOR = 1    # colour of the single initial cell
DEFAULT_MAX_STEPS = 3000  # a pattern still alive after this many generations
                          # counts as "infinite" (fitness 0)

# ----------------------------------------------------------------------------
# rule-case bookkeeping
# ----------------------------------------------------------------------------

def case_index(a: int, b: int, c: int, k: int = DEFAULT_K) -> int:
    """Index into the full rule table of neighbourhood (left, centre, right)."""
    return k * k * a + k * b + c


def index_case(i: int, k: int = DEFAULT_K):
    """Inverse of :func:`case_index`."""
    return i // (k * k), (i // k) % k, i % k


def symmetric_cases(k: int = DEFAULT_K):
    """The independent neighbourhoods of a left-right symmetric rule,
    i.e. all (a, b, c) with a <= c, as a list in a fixed order.
    Length  k * k * (k + 1) / 2  (75 for k = 5, 18 for k = 3)."""
    return [(a, b, c) for a in range(k) for b in range(k) for c in range(k)
            if a <= c]


class Rule:
    """A left-right symmetric k-colour nearest-neighbour CA rule.

    ``table`` is the full k**3-entry lookup table indexed by
    :func:`case_index`; the symmetry constraint is enforced on construction.
    """

    __slots__ = ("table", "k")

    def __init__(self, table, k: Optional[int] = None):
        table = np.ascontiguousarray(table, dtype=np.uint8)
        if k is None:
            k = int(round(len(table) ** (1.0 / 3.0)))
        if len(table) != k ** 3:
            raise ValueError(f"table has {len(table)} entries, expected k**3 = {k**3}")
        if table.max(initial=0) >= k:
            raise ValueError("table contains colours >= k")
        self.table = table
        self.k = k
        if not self.is_symmetric():
            raise ValueError("rule is not left-right symmetric; use Rule.from_cases")

    # -- constructors --------------------------------------------------------
    @classmethod
    def null(cls, k: int = DEFAULT_K) -> "Rule":
        """The rule that maps everything to white (the usual starting point)."""
        return cls(np.zeros(k ** 3, dtype=np.uint8), k)

    @classmethod
    def from_cases(cls, cases, k: int = DEFAULT_K) -> "Rule":
        """Build a rule from a mapping {(a, b, c): colour}; the mirror image
        (c, b, a) is filled in automatically.  Unspecified cases are white."""
        table = np.zeros(k ** 3, dtype=np.uint8)
        for (a, b, c), out in dict(cases).items():
            table[case_index(a, b, c, k)] = out
            table[case_index(c, b, a, k)] = out
        return cls(table, k)

    @classmethod
    def random(cls, rng=None, k: int = DEFAULT_K, fix_background: bool = True) -> "Rule":
        """A uniformly random symmetric rule (most of these grow forever)."""
        rng = np.random.default_rng(rng)
        cases = symmetric_cases(k)
        outs = rng.integers(0, k, size=len(cases))
        if fix_background:
            outs[0] = 0
        return cls.from_cases(dict(zip(cases, outs)), k)

    @classmethod
    def from_number(cls, number: int, k: int = DEFAULT_K) -> "Rule":
        """Inverse of :meth:`to_number`."""
        cases = symmetric_cases(k)
        digits = []
        for _ in cases:
            digits.append(number % k)
            number //= k
        if number:
            raise ValueError("rule number too large for this k")
        # digits were produced least-significant first = last case first
        return cls.from_cases(dict(zip(cases, reversed(digits))), k)

    # -- properties ----------------------------------------------------------
    def to_number(self) -> int:
        """Encode the rule as a single integer: the outputs of the independent
        cases (in :func:`symmetric_cases` order, first case = most significant
        digit) read as a base-k number.  A convenient way to save/share a
        genotype."""
        n = 0
        for (a, b, c) in symmetric_cases(self.k):
            n = n * self.k + int(self.table[case_index(a, b, c, self.k)])
        return n

    def cases(self):
        """List of ((a, b, c), output) for the independent cases."""
        return [((a, b, c), int(self.table[case_index(a, b, c, self.k)]))
                for (a, b, c) in symmetric_cases(self.k)]

    def is_symmetric(self) -> bool:
        k = self.k
        t = self.table.reshape(k, k, k)
        return bool(np.array_equal(t, t.transpose(2, 1, 0)))

    def __call__(self, a: int, b: int, c: int) -> int:
        return int(self.table[case_index(a, b, c, self.k)])

    def copy(self) -> "Rule":
        return Rule(self.table.copy(), self.k)

    def __eq__(self, other):
        return (isinstance(other, Rule) and self.k == other.k
                and np.array_equal(self.table, other.table))

    def __hash__(self):
        return hash((self.k, self.table.tobytes()))

    def __repr__(self):
        return f"Rule(k={self.k}, number={self.to_number()})"

    # -- mutation ------------------------------------------------------------
    def mutated(self, rng=None, n_mutations: int = 1,
                fix_background: bool = True) -> "Rule":
        """Return a copy with ``n_mutations`` distinct independent rule cases
        each changed to a different, uniformly random colour.  With
        ``fix_background`` the all-white case is never touched (if it were,
        the whole infinite line would change colour -> fitness 0 anyway)."""
        rng = np.random.default_rng(rng)
        cases = symmetric_cases(self.k)
        first = 1 if fix_background else 0
        picks = rng.choice(len(cases) - first, size=n_mutations, replace=False) + first
        table = self.table.copy()
        for p in picks:
            a, b, c = cases[p]
            i, j = case_index(a, b, c, self.k), case_index(c, b, a, self.k)
            new = (int(table[i]) + int(rng.integers(1, self.k))) % self.k
            table[i] = new
            table[j] = new
        return Rule(table, self.k)

    def distance(self, other: "Rule") -> int:
        """Number of independent cases on which two rules differ."""
        return sum(1 for (abc, x), (_, y) in zip(self.cases(), other.cases()) if x != y)


# ----------------------------------------------------------------------------
# running a rule: fast phenotype statistics
# ----------------------------------------------------------------------------

class Stats(NamedTuple):
    """Summary of a rulioorganism's phenotype.

    lifetime : generations before the pattern dies out (0 = never dies,
               i.e. periodic, or still alive after max_steps / max_width).
    width    : maximum width reached (number of cells between the outermost
               non-white cells, inclusive).
    counts   : tuple of length k; counts[c] = number of cells of colour c
               over the whole history.  counts[0] counts the white cells
               *inside* each row's non-white extent ("voids").
    """
    lifetime: int
    width: int
    counts: tuple

    @property
    def finite(self) -> bool:
        return self.lifetime > 0

    @property
    def cells(self) -> int:
        """Total number of non-white cells."""
        return int(sum(self.counts[1:]))

    @property
    def aspect(self) -> float:
        """width / lifetime (0 for infinite patterns)."""
        return self.width / self.lifetime if self.lifetime else 0.0

    @property
    def color_fractions(self) -> np.ndarray:
        """Fraction of non-white cells of each colour (index 0 is always 0)."""
        f = np.array(self.counts, dtype=float)
        f[0] = 0.0
        s = f.sum()
        return f / s if s else f


@_njit(cache=True, nogil=True)
def _stats_kernel(table, k, seed, max_steps, max_width):
    """Simulate the right half of the (mirror-symmetric) pattern.

    Returns (lifetime, max_width, counts) with lifetime = 0 for patterns that
    never die.  Periodicity is detected exactly by hashing each row; a
    growing pattern is declared infinite after max_steps generations or
    when it becomes wider than max_width.
    """
    kk = k * k
    cap = max_steps + 4
    cur = np.zeros(cap, np.uint8)   # cur[i] = cell at position +i (and -i)
    nxt = np.zeros(cap, np.uint8)
    cur[0] = seed
    hi = 0                          # outermost non-white position
    counts = np.zeros(k, np.int64)
    counts[seed] += 1
    widest = 1
    seen = {np.int64(0)}            # hashes of all previous rows
    for t in range(1, max_steps + 1):
        # --- periodicity check on the current row (generation t-1) --------
        h = np.int64(-3750763034362895579)          # FNV-1a, 64 bit
        for i in range(hi + 1):
            h = (h ^ np.int64(cur[i])) * np.int64(1099511628211)
        h ^= np.int64(hi)
        if h in seen:
            return 0, widest, counts
        seen.add(h)
        # --- next generation on positions 0 .. hi+1 -----------------------
        nhi = -1
        for i in range(hi + 2):
            left = cur[i - 1] if i > 0 else cur[1]   # mirror symmetry at 0
            v = table[kk * left + k * cur[i] + cur[i + 1]]
            nxt[i] = v
            if v != 0:
                nhi = i
        for i in range(hi + 1):                      # keep buffers clean
            cur[i] = 0
        if nhi < 0:
            return t, widest, counts                 # died: t generations lived
        w = 2 * nhi + 1
        if w > widest:
            widest = w
        if w > max_width:
            return 0, widest, counts
        counts[nxt[0]] += 1
        for i in range(1, nhi + 1):
            counts[nxt[i]] += 2
        cur, nxt = nxt, cur
        hi = nhi
    return 0, widest, counts


def _stats_numpy(table, k, seed, max_steps, max_width):
    """Pure-numpy version of :func:`_stats_kernel` (full row, same results)."""
    kk = k * k
    row = np.array([seed], dtype=np.uint8)
    counts = np.zeros(k, dtype=np.int64)
    counts[seed] += 1
    widest = 1
    seen = set()
    for t in range(1, max_steps + 1):
        key = row.tobytes()
        if key in seen:
            return 0, widest, counts
        seen.add(key)
        padded = np.zeros(row.size + 4, dtype=np.int64)
        padded[2:-2] = row
        new = table[kk * padded[:-2] + k * padded[1:-1] + padded[2:]]
        nz = np.flatnonzero(new)
        if nz.size == 0:
            return t, widest, counts
        row = new[nz[0]:nz[-1] + 1]
        if row.size > widest:
            widest = row.size
        if row.size > max_width:
            return 0, widest, counts
        counts += np.bincount(row, minlength=k)
    return 0, widest, counts


def stats(rule: Rule, max_steps: int = DEFAULT_MAX_STEPS, max_width: Optional[int] = None,
          seed_color: int = DEFAULT_SEED_COLOR, engine: str = "auto") -> Stats:
    """Run the rule from a single seed and return its :class:`Stats`."""
    if max_width is None:
        max_width = 2 * max_steps + 1
    if engine == "auto":
        engine = "numba" if HAVE_NUMBA else "numpy"
    if engine == "numba":
        if not HAVE_NUMBA:
            raise RuntimeError("numba is not installed")
        life, width, counts = _stats_kernel(rule.table, rule.k, seed_color,
                                            max_steps, max_width)
    elif engine == "numpy":
        life, width, counts = _stats_numpy(rule.table, rule.k, seed_color,
                                           max_steps, max_width)
    else:
        raise ValueError("engine must be 'auto', 'numba' or 'numpy'")
    return Stats(int(life), int(width), tuple(int(c) for c in counts))


def lifetime(rule: Rule, **kwargs) -> int:
    """Lifetime of the pattern grown from a single seed (0 = infinite)."""
    return stats(rule, **kwargs).lifetime


# ----------------------------------------------------------------------------
# running a rule: full space-time history (for pictures and dissection)
# ----------------------------------------------------------------------------

def grow(rule: Rule, steps: Optional[int] = None, initial=None,
         seed_color: int = DEFAULT_SEED_COLOR, max_steps: int = DEFAULT_MAX_STEPS,
         max_width: Optional[int] = None) -> np.ndarray:
    """Space-time history of a rulioorganism as a (generations, width) uint8
    array, one generation per row, trimmed to the bounding box.

    steps   : number of generations to compute after the initial one.
              None = run until the pattern dies (or max_steps is reached).
    initial : optional 1-D array of colours to start from instead of a
              single seed cell -- e.g. a patch cut out of a larger organism,
              to see how a "part" behaves in isolation.
    """
    k, table, kk = rule.k, rule.table, rule.k * rule.k
    if initial is None:
        row = np.array([seed_color], dtype=np.uint8)
        offset = 0
    else:
        row = np.asarray(initial, dtype=np.uint8).ravel()
        nz = np.flatnonzero(row)
        if nz.size == 0:
            return np.zeros((1, max(row.size, 1)), dtype=np.uint8)
        offset = int(nz[0])
        row = row[nz[0]:nz[-1] + 1]
    if max_width is None:
        max_width = 2 * max_steps + 1
    n_steps = max_steps if steps is None else steps
    rows = [(offset, row)]
    for _ in range(n_steps):
        padded = np.zeros(row.size + 4, dtype=np.int64)
        padded[2:-2] = row
        new = table[kk * padded[:-2] + k * padded[1:-1] + padded[2:]]
        nz = np.flatnonzero(new)
        if nz.size == 0:
            break
        offset = offset - 1 + int(nz[0])
        row = new[nz[0]:nz[-1] + 1]
        rows.append((offset, row))
        if row.size > max_width:
            break
    lo = min(o for o, _ in rows)
    hi = max(o + r.size for o, r in rows)
    hist = np.zeros((len(rows), hi - lo), dtype=np.uint8)
    for t, (o, r) in enumerate(rows):
        hist[t, o - lo:o - lo + r.size] = r
    return hist


def used_cases(rule: Rule, steps: int, seed_color: int = DEFAULT_SEED_COLOR) -> set:
    """Indices (into the full k**3 table) of the rule cases actually applied
    during the first ``steps`` generations of development.  Wolfram's
    "conservation of development": early development is conserved along a
    lineage because only this subset of the genome is expressed early."""
    k, kk = rule.k, rule.k * rule.k
    hist = grow(rule, steps=steps, seed_color=seed_color)
    used = set()
    for row in hist[:steps]:
        padded = np.zeros(row.size + 4, dtype=np.int64)
        padded[2:-2] = row
        idx = kk * padded[:-2] + k * padded[1:-1] + padded[2:]
        used.update(int(i) for i in np.unique(idx))
    used.discard(0)   # the background case is used everywhere, trivially
    return used


# ----------------------------------------------------------------------------
# fitness functions  (Stats -> number; 0 means "reject")
# ----------------------------------------------------------------------------

def fit_lifetime(s: Stats) -> float:
    """Wolfram's standard fitness: live as long as possible (then die)."""
    return s.lifetime


def fit_width(s: Stats) -> float:
    """Be as wide as possible (among patterns that eventually die)."""
    return s.width if s.finite else 0


def fit_wide(s: Stats) -> float:
    """Maximise width / height  (wide, flat organisms)."""
    return s.width / s.lifetime if s.finite else 0


def fit_tall(s: Stats) -> float:
    """Maximise height / width  (tall, thin organisms)."""
    return s.lifetime / s.width if s.finite else 0


def fit_cells(s: Stats) -> float:
    """Maximise the total number of non-white cells."""
    return s.cells if s.finite else 0


class _ColorFitness:
    """Maximise the number of cells of one colour (a class, so that instances
    can be pickled and sent to worker processes by ``evolve_many``)."""

    def __init__(self, c: int):
        self.c = c
        self.__name__ = f"fit_color_{c}"

    def __call__(self, s: Stats) -> float:
        return s.counts[self.c] if s.finite else 0


def fit_color(c: int) -> Callable[[Stats], float]:
    """Factory: maximise the number of cells of colour ``c``."""
    return _ColorFitness(c)


# ----------------------------------------------------------------------------
# adaptive evolution
# ----------------------------------------------------------------------------

@dataclass
class Evolution:
    """Record of one run of adaptive evolution (one lineage)."""
    rule: Rule                       # final rule (genotype)
    fitness: float                   # its fitness
    attempts: np.ndarray             # fitness of the mutant tried at each step
    best: np.ndarray                 # fitness of the current organism after each step
    lineage: List[tuple]             # (step, fitness, Rule) for every accepted mutation
    breakthroughs: List[tuple]       # subset of lineage with a strict fitness increase
    fitness_name: str = "lifetime"
    params: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return len(self.attempts)

    @property
    def breakthrough_rules(self) -> List[Rule]:
        return [r for _, _, r in self.breakthroughs]

    def extend(self, more: "Evolution") -> "Evolution":
        """Append a continuation run (one started with ``start=self.rule``)
        so that the whole history reads as a single lineage."""
        off = self.steps
        lineage = self.lineage + [(s + off, f, r) for s, f, r in more.lineage[1:]]
        bts = self.breakthroughs + [(s + off, f, r) for s, f, r in more.breakthroughs[1:]]
        params = dict(self.params); params["steps"] = off + more.steps
        return Evolution(more.rule, more.fitness, np.concatenate([self.attempts, more.attempts]),
                         np.concatenate([self.best, more.best]), lineage, bts,
                         self.fitness_name, params)

    def summary(self) -> str:
        n_acc = len(self.lineage) - 1
        n_inf = int(np.sum(self.attempts == 0))
        return (f"{self.steps} steps of adaptive evolution ({self.fitness_name}): "
                f"final fitness {self.fitness:g}, {len(self.breakthroughs) - 1} breakthroughs, "
                f"{n_acc} accepted mutations ({n_acc - len(self.breakthroughs) + 1} neutral), "
                f"{n_inf} mutants grew forever.  Final rule number {self.rule.to_number()}")


def evolve(steps: int = 1000, fitness: Callable[[Stats], float] = fit_lifetime,
           start: Optional[Rule] = None, seed=None, n_mutations: int = 1,
           k: int = DEFAULT_K, max_steps: int = DEFAULT_MAX_STEPS,
           max_width: Optional[int] = None, seed_color: int = DEFAULT_SEED_COLOR,
           fix_background: bool = True, progress: bool = False,
           evaluate: Optional[Callable] = None,
           callback: Optional[Callable[[int, float, int], bool]] = None) -> Evolution:
    """Run Wolfram's adaptive-evolution loop for ``steps`` mutation attempts.

    Starting from ``start`` (default: the null rule), each step mutates
    ``n_mutations`` rule cases at random; the mutant replaces the current
    organism iff fitness(mutant) >= fitness(current).  ``seed`` makes the run
    reproducible (any numpy random seed / Generator).

    ``evaluate`` (rule -> Stats-like object) replaces the default
    ``stats(rule, max_steps=..., ...)``; this is how the 2-D module reuses
    the loop with its own rules and phenotype statistics.

    ``callback(step, fitness, n_breakthroughs)`` is called after every step
    (for progress bars); returning True stops the run early.
    """
    rng = np.random.default_rng(seed)
    rule = start.copy() if start is not None else Rule.null(k)
    if evaluate is None:
        kw = dict(max_steps=max_steps, max_width=max_width, seed_color=seed_color)

        def evaluate(r):
            return stats(r, **kw)
    cur = fitness(evaluate(rule))
    attempts = np.empty(steps, dtype=float)
    best = np.empty(steps, dtype=float)
    lineage = [(0, cur, rule.copy())]
    breakthroughs = [(0, cur, rule.copy())]
    for step in range(1, steps + 1):
        cand = rule.mutated(rng, n_mutations, fix_background)
        f = fitness(evaluate(cand))
        attempts[step - 1] = f
        if f >= cur:
            if f > cur:
                breakthroughs.append((step, f, cand.copy()))
                if progress:
                    print(f"step {step:6d}: fitness {cur:g} -> {f:g}")
            lineage.append((step, f, cand.copy()))
            rule, cur = cand, f
        best[step - 1] = cur
        if callback is not None and callback(step, cur, len(breakthroughs) - 1):
            attempts, best = attempts[:step], best[:step]
            break
    return Evolution(rule, cur, attempts, best, lineage, breakthroughs,
                     getattr(fitness, "__name__", "fitness").replace("fit_", ""),
                     dict(steps=steps, seed=seed, n_mutations=n_mutations, k=k,
                          max_steps=max_steps, max_width=max_width,
                          seed_color=seed_color, fix_background=fix_background))


def _evolve_job(args):
    seed, kwargs = args
    return evolve(seed=seed, **kwargs)


def evolve_many(n_runs: int, steps: int = 1000, seed=0, n_jobs: int = 1,
                **kwargs) -> List[Evolution]:
    """``n_runs`` independent runs (different random seeds), optionally in
    parallel processes.  Fitness functions must be picklable (module-level)
    when n_jobs > 1."""
    seeds = np.random.SeedSequence(seed).spawn(n_runs)
    jobs = [(s, dict(steps=steps, **kwargs)) for s in seeds]
    if n_jobs == 1:
        return [_evolve_job(j) for j in jobs]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        return list(ex.map(_evolve_job, jobs))


# ----------------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------------

# white, red, yellow, green, purple, ... (the colours Wolfram names in the essay)
PALETTE = ["#FFFFFF", "#D7302A", "#F2C21B", "#4FA64F", "#7B4DB3",
           "#2E7BD6", "#F07E1A", "#2BB5B5", "#8C5A2B", "#E350A8"]


def colormap(k: int = DEFAULT_K):
    """ListedColormap for k colours (index 0 = white background)."""
    from matplotlib.colors import ListedColormap
    if k <= len(PALETTE):
        return ListedColormap(PALETTE[:k])
    from matplotlib.colors import hsv_to_rgb
    extra = hsv_to_rgb(np.stack([np.linspace(0, 0.9, k - 1), np.full(k - 1, 0.85),
                                 np.full(k - 1, 0.85)], axis=1))
    return ListedColormap([PALETTE[0]] + [tuple(c) for c in extra])


def plot_organism(rule_or_history, ax=None, title=None, max_steps: int = DEFAULT_MAX_STEPS,
                  seed_color: int = DEFAULT_SEED_COLOR, k: Optional[int] = None,
                  max_inches: float = 8.0, frame: bool = False, min_size: int = 12):
    """Draw a rulioorganism (a Rule, or a history array from :func:`grow`).
    Tiny patterns are padded to ``min_size`` cells so they stay tiny."""
    import matplotlib.pyplot as plt
    if isinstance(rule_or_history, Rule):
        hist = grow(rule_or_history, max_steps=max_steps, seed_color=seed_color)
        k = rule_or_history.k
    else:
        hist = np.asarray(rule_or_history)
        k = k or max(DEFAULT_K, int(hist.max()) + 1)
    rows, cols = hist.shape
    if rows < min_size or cols < min_size:          # keep the seed at the top
        pad_w = max(min_size - cols, 0)
        hist = np.pad(hist, ((0, max(min_size - rows, 0)), (pad_w // 2, pad_w - pad_w // 2)))
        rows, cols = hist.shape
    if ax is None:
        scale = max_inches / max(rows, cols)
        _, ax = plt.subplots(figsize=(max(cols * scale, 1.0), max(rows * scale, 1.0)))
    ax.imshow(hist, cmap=colormap(k), vmin=0, vmax=k - 1,
              interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if not frame:
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def plot_fitness(ev: Evolution, ax=None, log: bool = False):
    """Fitness of every mutant tried (red dots) and of the current organism
    (black steps), as in Wolfram's figures."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(1, ev.steps + 1)
    ax.plot(x, ev.attempts, ".", color="#D7302A", ms=3, alpha=0.6, label="mutant tried")
    ax.step(x, ev.best, where="post", color="black", lw=1.2, label="current organism")
    for s, f, _ in ev.breakthroughs[1:]:
        ax.plot(s, f, "o", mfc="none", mec="black", ms=6)
    ax.set_xlabel("adaptive evolution step")
    ax.set_ylabel(ev.fitness_name)
    if log:
        ax.set_yscale("symlog", linthresh=1)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return ax


def plot_gallery(rules: Sequence[Rule], ncols: int = 6, titles: Optional[Sequence] = None,
                 max_steps: int = DEFAULT_MAX_STEPS, cell: float = 2.2, **kw):
    """Grid of organisms."""
    import matplotlib.pyplot as plt
    n = len(rules)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * nrows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            plot_organism(rules[i], ax=ax, max_steps=max_steps,
                          title=None if titles is None else titles[i], **kw)
    fig.tight_layout()
    return fig


def plot_breakthroughs(ev: Evolution, max_show: int = 12, ncols: Optional[int] = None, **kw):
    """The "fossil record": the organism at each fitness breakthrough.
    If there are more than ``max_show`` breakthroughs, a subsample (always
    including the last one) is shown."""
    bts = ev.breakthroughs[1:] or ev.breakthroughs
    if len(bts) > max_show:
        idx = np.unique(np.linspace(0, len(bts) - 1, max_show).round().astype(int))
        bts = [bts[i] for i in idx]
    titles = [f"step {s}: {ev.fitness_name} {f:g}" for s, f, _ in bts]
    return plot_gallery([r for _, _, r in bts], ncols=ncols or min(6, len(bts)),
                        titles=titles, **kw)


def plot_development(ev: Evolution, steps: int = 30, max_show: int = 12, cell: float = 1.8):
    """Early development (first ``steps`` generations) of each breakthrough
    organism side by side -- Wolfram's "conservation of development"."""
    import matplotlib.pyplot as plt
    bts = ev.breakthroughs[1:] or ev.breakthroughs
    if len(bts) > max_show:
        idx = np.unique(np.linspace(0, len(bts) - 1, max_show).round().astype(int))
        bts = [bts[i] for i in idx]
    hists = [grow(r, steps=steps, seed_color=ev.params.get("seed_color", DEFAULT_SEED_COLOR))
             for _, _, r in bts]
    w = max(h.shape[1] for h in hists)
    fig, axes = plt.subplots(1, len(hists), figsize=(cell * len(hists), cell * 1.4), squeeze=False)
    for ax, h, (s, f, r) in zip(axes[0], hists, bts):
        pad = np.zeros((steps + 1, w), dtype=np.uint8)
        lo = (w - h.shape[1]) // 2
        pad[:h.shape[0], lo:lo + h.shape[1]] = h[:steps + 1]
        plot_organism(pad, ax=ax, k=ev.rule.k, title=f"step {s}\n{ev.fitness_name} {f:g}")
    fig.tight_layout()
    return fig


def plot_rule(rule: Rule, ax=None, ncols: int = 15, cell: float = 0.33):
    """Draw the rule as Wolfram-style icons: each independent case shows the
    three-cell neighbourhood above its one-cell outcome."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    k = rule.k
    cases = rule.cases()
    nrows = math.ceil(len(cases) / ncols)
    gap = k                       # extra index for the grey background
    img = np.full((nrows * 3, ncols * 4), gap, dtype=np.int64)
    for n, ((a, b, c), out) in enumerate(cases):
        r, col = divmod(n, ncols)
        y, x = 3 * r, 4 * col
        img[y, x:x + 3] = (a, b, c)
        img[y + 1, x + 1] = out
    colors = list(colormap(k).colors) + ["#E8E8E8"]
    if ax is None:
        _, ax = plt.subplots(figsize=(ncols * 4 * cell, nrows * 3 * cell))
    ax.imshow(img, cmap=ListedColormap(colors), vmin=0, vmax=k, interpolation="nearest")
    # thin cell borders so white cells are visible
    for y in range(nrows):
        for x in range(ncols):
            if y * ncols + x < len(cases):
                ax.add_patch(plt.Rectangle((4 * x - 0.5, 3 * y - 0.5), 3, 1, fill=False,
                                           ec="#BBBBBB", lw=0.5))
                ax.add_patch(plt.Rectangle((4 * x + 0.5, 3 * y + 0.5), 1, 1, fill=False,
                                           ec="#BBBBBB", lw=0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


__all__ = [
    "DEFAULT_K", "DEFAULT_MAX_STEPS", "DEFAULT_SEED_COLOR", "HAVE_NUMBA",
    "Rule", "Stats", "Evolution", "symmetric_cases", "case_index", "index_case",
    "stats", "lifetime", "grow", "used_cases",
    "fit_lifetime", "fit_width", "fit_wide", "fit_tall", "fit_cells", "fit_color",
    "evolve", "evolve_many",
    "PALETTE", "colormap", "plot_organism", "plot_fitness", "plot_gallery",
    "plot_breakthroughs", "plot_development", "plot_rule",
]
