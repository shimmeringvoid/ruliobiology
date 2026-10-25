#!/usr/bin/env python3
"""
Evolve two-dimensional rulioorganisms and look at them, including as a
rotatable 3-D space-time volume.

    python examples/demo2d.py                         # 4 lineages x 10000 steps, k=4, von Neumann
    python examples/demo2d.py --runs 12 --steps 20000 --jobs 4
    python examples/demo2d.py --k 3 --neighborhood moore
    python examples/demo2d.py --show

Written to --out (default ./figures2d):
    gallery2d.png     one projection per lineage (colour each cell had when last alive)
    filmstrip.png     the longest-lived organism, 12 generations across its life
    projections.png   the same organism: last colour / birth time / generations alive
    fitness2d.png     fitness trajectory of that lineage
    organism3d.html   the space-time volume in a rotatable, translucent 3-D viewer
"""
import argparse
import os
import time

import numpy as np

import ruliobiology as rb   # pip install -e .  (or run from the repo root)
from ruliobiology import ca2d as rb2
from ruliobiology.viewer3d import write_viewer


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=4, help="independent lineages (default 4)")
    p.add_argument("--steps", type=int, default=10000, help="mutation attempts per lineage")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--k", type=int, default=rb2.DEFAULT_K, help="number of colours (default 4)")
    p.add_argument("--neighborhood", choices=sorted(rb2.NEIGHBORHOODS), default=rb2.DEFAULT_NEIGHBORHOOD)
    p.add_argument("--max-steps", type=int, default=rb2.DEFAULT_MAX_STEPS,
                   help="generations after which a pattern counts as infinite")
    p.add_argument("--max-width", type=int, default=rb2.DEFAULT_MAX_WIDTH,
                   help="bounding-box side beyond which a pattern counts as infinite")
    p.add_argument("--jobs", type=int, default=1, help="processes to run lineages in parallel")
    p.add_argument("--out", default="figures2d")
    p.add_argument("--dpi", type=int, default=130)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    kw = dict(k=args.k, neighborhood=args.neighborhood, max_steps=args.max_steps,
              max_width=args.max_width)

    def save(fig, name):
        path = os.path.join(args.out, name)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print("  wrote", path)

    print(f"Evolving {args.runs} lineage(s) x {args.steps} steps: k={args.k}, {args.neighborhood} "
          f"neighbourhood, {len(rb2.valid_cases(args.k, len(rb2.NEIGHBORHOODS[args.neighborhood])))} "
          f"rule cases ({'numba' if rb.HAVE_NUMBA else 'numpy -- pip install numba for speed'})")
    t0 = time.time()
    evs = rb2.evolve2d_many(args.runs, steps=args.steps, seed=args.seed, n_jobs=args.jobs, **kw)
    print(f"  done in {time.time() - t0:.1f}s")
    for i, e in enumerate(evs):
        s = rb2.stats2d(e.rule, max_steps=args.max_steps, max_width=args.max_width)
        print(f"  lineage {i}: lifetime {s.lifetime:4d}  width {s.width:3d}  max area {s.max_area:4d}  "
              f"cells {s.cells:7,d}  breakthroughs {len(e.breakthroughs) - 1}")

    order = np.argsort([-e.fitness for e in evs])
    save(rb2.plot_gallery2d([evs[i].rule for i in order], ncols=min(6, len(evs)),
                            titles=[f"lifetime {evs[i].fitness:g}" for i in order],
                            max_steps=args.max_steps, max_width=args.max_width), "gallery2d.png")

    best = evs[order[0]]
    vol = rb2.grow2d(best.rule, max_steps=args.max_steps, max_width=args.max_width)
    save(rb2.plot_filmstrip(vol, n=12, k=args.k), "filmstrip.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, mode in zip(axes, ["last_color", "birth", "count"]):
        rb2.plot_projection(vol, mode, ax=ax, k=args.k)
    save(fig, "projections.png")
    save(rb.plot_fitness(best).figure, "fitness2d.png")

    path = os.path.join(args.out, "organism3d.html")
    write_viewer(vol, path, title=f"Lineage {int(order[0])}",
                 caption=f"{args.k} colours, {args.neighborhood} neighbourhood, {args.steps:,} steps "
                         f"of adaptive evolution for lifetime. Rule number {best.rule.to_number()}")
    print("  wrote", path, "-- open it in a browser")
    print(f"  best rule: {best.rule!r}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
