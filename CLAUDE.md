# CLAUDE.md — working notes for this repository

Read this before changing anything. It records what the project is, the
decisions already made, and the habits that keep it correct.

## What this is

A Python implementation of Stephen Wolfram's *ruliobiology* model (adaptive
evolution of simple cellular-automaton rules, fitness = lifetime), in 1-D
(exactly his setup) and 2-D (this repo's extension), plus a rotatable 3-D
viewer for 2-D organisms and a Streamlit GUI. The README is the user-facing
document; keep it accurate when behaviour changes.

Reference: https://writings.stephenwolfram.com/2026/10/endless-forms-most-beautiful-and-most-wonderful-ruliobiology-and-the-theoretical-foundations-of-biological-form/

## Layout

- `ruliobiology/ca1d.py` — 1-D model. Owns the shared evolution loop
  (`evolve`, with `evaluate` and `callback` hooks), the `Evolution` record,
  the fitness functions and the plotting helpers.
- `ruliobiology/ca2d.py` — 2-D model (outer-totalistic rules, von Neumann or
  Moore). Reuses `ca1d.evolve` through `evaluate=`; never duplicate the loop.
- `ruliobiology/viewer3d.py` — `write_viewer()`: one self-contained HTML page
  (three.js r128 from cdnjs; the JS lives in `_SCRIPT`, the CSS in `_HEAD`).
- `app.py` — the GUI (`streamlit run app.py`). All state lives in
  `st.session_state` under per-model keys (`1d_ev`, `2d_ev`, ...).
- `examples/` — command-line demos. `tests/` — pytest. `docs/images/` — README figures.

## Environment and commands

```bash
conda activate ruliobiology        # from environment.yml; pip install -e . once
pytest -q                          # must pass before any commit (26 tests, ~6 s)
python examples/demo1d.py          # quick visual sanity check -> figures/
streamlit run app.py
```

Dependencies are numpy + matplotlib; numba, scipy (tests only) and streamlit
are optional. Do not add a dependency without adding it to `environment.yml`,
`requirements.txt` and `pyproject.toml` together.

## Invariants to preserve

- **Two engines, one answer.** Every phenotype statistic has a numba kernel
  (`_stats_kernel`, `_stats2d_kernel`) and a pure-numpy fallback
  (`_stats_numpy`, `_stats2d_numpy`). They must return identical results;
  `test_engines_agree` / `test_numba_and_numpy_engines_agree` check this. If
  you change what is measured, change both and extend the test.
- **`grow` agrees with `stats`.** The picture and the numbers come from
  different code paths; `test_grow_matches_stats` keeps them consistent.
- **Symmetry.** 1-D rules are left–right symmetric (`Rule.from_cases` fills in
  mirrors; the kernel simulates the right half). 2-D rules are outer-totalistic,
  hence D4-symmetric (the kernel simulates one quadrant). Anything that breaks
  symmetry breaks both kernels.
- **Infinite patterns have fitness 0.** Periodicity is detected exactly by
  hashing rows/frames; growth is cut off by `max_steps` (and `max_width` in
  2-D). In 2-D the width cap is what keeps runs fast (cost ~ t³ for light-speed
  growth); don't remove it.
- **Rule numbers are a file format.** `to_number()` / `from_number()` read the
  independent cases in the fixed order of `symmetric_cases()` /
  `valid_cases()`. Changing that order silently invalidates every saved
  organism, so don't.
- **Fitness functions must be picklable** (module-level functions or instances
  of module-level classes, like `_ColorFitness`) because `evolve_many` sends
  them to worker processes. Lambdas are fine only with `n_jobs=1`.
- **Lifetime counts non-white generations**: the null rule has lifetime 1 and a
  picture has exactly `lifetime` rows/frames.

## Known behaviour that is not a bug

- Roughly one 1-D lineage in six never leaves lifetime 1 in 1000 steps
  (neutral drift of the unexpressed genome; see README).
- Large 2-D genomes (k=4 Moore, 660 cases) evolve very slowly; k=4 von Neumann
  (140 cases) is the default for that reason.
- The viewer page needs internet access for three.js; it shows a message
  instead of a blank pane when the library fails to load.

## Working conventions

- Keep the code in the style of the existing modules: plain numpy functions,
  docstrings that say what a thing is for, no classes beyond `Rule`/`Rule2D`,
  `Stats`/`Stats2D`, `Evolution`.
- Figures: matplotlib, `Agg`-safe (never call `plt.show()` inside library
  code); every plot helper returns an Axes or Figure.
- Commit messages describe the behaviour change, not the files touched.
- Ask before deleting files or rewriting history; prefer additive changes.
- When a change affects what users see (defaults, outputs, GUI), update the
  README in the same commit.

## Ideas not yet done

- A GPU-batched evaluator (CUDA via numba.cuda or CuPy) that grows many
  mutants or lineages at once; the single-lineage loop itself is sequential.
- Exact convergence studies via multiway graphs (as in Wolfram's 2024 essays)
  for small rule spaces.
- Persisting a "zoo" of organisms to disk from the GUI.
