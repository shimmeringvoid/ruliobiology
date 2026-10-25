#!/usr/bin/env python3
"""
Reproduce the basic 1-D ruliobiology experiments from Wolfram's essay and
save the figures as PNG files.

    python examples/demo1d.py                       # one lineage, 1000 steps, seed 1
    python examples/demo1d.py --steps 2000 --seed 7 --fitness tall
    python examples/demo1d.py --runs 36 --jobs 4    # + a gallery of 36 independent lineages
    python examples/demo1d.py --show                # open the figures interactively

Figures written to --out (default ./figures):
    fitness.png        fitness of every mutant tried, and of the current organism
    fossil_record.png  the organism at each fitness breakthrough
    organism.png       the final rulioorganism (phenotype)
    rule.png           its rule, as Wolfram-style case icons (genotype)
    development.png    first 30 generations of each breakthrough organism
    gallery.png        (with --runs) final organisms of independent lineages
    side_effects.png   (with --runs) lifetime vs width: traits nobody selected for
"""
import argparse
import os
import time

import numpy as np

import ruliobiology as rb   # pip install -e .  (or run from the repo root)

FITNESS = {
    "lifetime": rb.fit_lifetime,
    "width": rb.fit_width,
    "wide": rb.fit_wide,
    "tall": rb.fit_tall,
    "cells": rb.fit_cells,
    "red": rb.fit_color(1),
    "yellow": rb.fit_color(2),
    "green": rb.fit_color(3),
    "purple": rb.fit_color(4),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=1000, help="mutation attempts (default 1000)")
    p.add_argument("--seed", type=int, default=1, help="random seed of the lineage")
    p.add_argument("--fitness", choices=sorted(FITNESS), default="lifetime")
    p.add_argument("--k", type=int, default=rb.DEFAULT_K, help="number of colours (default 5)")
    p.add_argument("--mutations", type=int, default=1, help="rule cases changed per step")
    p.add_argument("--max-steps", type=int, default=rb.DEFAULT_MAX_STEPS,
                   help="generations after which a pattern counts as infinite")
    p.add_argument("--runs", type=int, default=0,
                   help="also evolve this many independent lineages for a gallery")
    p.add_argument("--jobs", type=int, default=1, help="processes for --runs")
    p.add_argument("--out", default="figures", help="output directory")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true", help="also display the figures")
    args = p.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    fitness = FITNESS[args.fitness]
    kw = dict(k=args.k, n_mutations=args.mutations, max_steps=args.max_steps)

    def save(fig, name):
        path = os.path.join(args.out, name)
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print("  wrote", path)

    # ------------------------------------------------------------ one lineage
    print(f"Evolving one lineage: {args.steps} steps, seed {args.seed}, fitness = {args.fitness}"
          f" ({'numba' if rb.HAVE_NUMBA else 'numpy -- pip install numba for ~5-10x speed'})")
    t0 = time.time()
    ev = rb.evolve(steps=args.steps, fitness=fitness, seed=args.seed, progress=True, **kw)
    print(f"  done in {time.time() - t0:.1f}s")
    print(" ", ev.summary())
    s = rb.stats(ev.rule, max_steps=args.max_steps)
    print(f"  final organism: lifetime {s.lifetime}, width {s.width}, {s.cells} cells, "
          f"colour fractions {np.round(s.color_fractions[1:], 3)}")

    save(rb.plot_fitness(ev).figure, "fitness.png")
    save(rb.plot_breakthroughs(ev, max_steps=args.max_steps), "fossil_record.png")
    save(rb.plot_organism(ev.rule, max_steps=args.max_steps, max_inches=10).figure, "organism.png")
    save(rb.plot_rule(ev.rule).figure, "rule.png")
    save(rb.plot_development(ev, steps=30), "development.png")

    # ------------------------------------------------------ many lineages
    if args.runs:
        print(f"Evolving {args.runs} independent lineages ({args.jobs} process(es))...")
        t0 = time.time()
        evs = rb.evolve_many(args.runs, steps=args.steps, seed=args.seed, n_jobs=args.jobs,
                             fitness=fitness, **kw)
        print(f"  done in {time.time() - t0:.1f}s")
        fits = np.array([e.fitness for e in evs])
        print(f"  final {args.fitness}: median {np.median(fits):g}, max {fits.max():g}, "
              f"{int((fits <= 1).sum())} lineage(s) never got going")
        order = np.argsort(-fits)
        rules = [evs[i].rule for i in order]
        titles = [f"{args.fitness} {fits[i]:g}" for i in order]
        ncols = int(np.ceil(np.sqrt(len(rules) * 1.5)))
        save(rb.plot_gallery(rules, ncols=ncols, titles=titles, max_steps=args.max_steps),
             "gallery.png")

        # traits that were never selected for: lifetime vs width ("side effects")
        st = [rb.stats(e.rule, max_steps=args.max_steps) for e in evs]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter([x.lifetime for x in st], [x.width for x in st], s=14, color="#D7302A")
        ax.set_xlabel("lifetime"); ax.set_ylabel("max width")
        ax.set_title(f"{len(evs)} lineages, {args.steps} steps, selected for {args.fitness}",
                     fontsize=9)
        save(fig, "side_effects.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
