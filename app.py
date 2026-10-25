#!/usr/bin/env python3
"""
app.py -- a browser GUI for the ruliobiology package.

    streamlit run app.py

Evolve 1-D or 2-D rulioorganisms with a progress bar, inspect the fitness
trajectory, the "fossil record" of breakthroughs, the organism, its genome
and its early development; rotate 2-D organisms in 3-D; evolve galleries of
independent lineages; save and reload organisms by rule number.
"""
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

import ruliobiology as rb
from ruliobiology import ca1d, ca2d
from ruliobiology.viewer3d import write_viewer

WOLFRAM = ("https://writings.stephenwolfram.com/2026/10/endless-forms-most-beautiful-and-most-"
           "wonderful-ruliobiology-and-the-theoretical-foundations-of-biological-form/")
COLOR_NAMES = ["white", "red", "yellow", "green", "purple", "blue", "orange", "teal", "brown", "pink"]

st.set_page_config(page_title="Ruliobiology", page_icon="🧬", layout="wide")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def fitness_options(model: str, k: int) -> dict:
    opts = {"lifetime": rb.fit_lifetime}
    if model == "2d":
        opts["largest area (at any moment)"] = rb.fit_area
        opts["volume (cells over the whole life)"] = rb.fit_volume
    else:
        opts["cells (over the whole life)"] = rb.fit_cells
    opts.update({"width": rb.fit_width, "wide: width / lifetime": rb.fit_wide,
                 "tall: lifetime / width": rb.fit_tall})
    for c in range(1, k):
        opts[f"most {COLOR_NAMES[c]} cells"] = rb.fit_color(c)
    return opts


def png(fig, dpi=130) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show(fig, name: str, dpi=130):
    """Render a figure, add a download button, release it."""
    data = png(fig, dpi)
    st.image(data)
    st.download_button(f"Download {name}", data, file_name=name, mime="image/png", key=f"dl_{name}_{time.time_ns()}")


def rule_from_number(model: str, text: str, k: int, neighborhood: str):
    n = int(text.strip())
    if model == "2d":
        return rb.Rule2D.from_number(n, k, neighborhood)
    return rb.Rule.from_number(n, k)


def run_evolution(model, steps, fitness, start, seed, n_mutations, k, neighborhood,
                  max_steps, max_width):
    bar = st.progress(0.0)
    status = st.empty()
    t0 = time.time()

    def cb(step, fit, nb):
        if step % 20 == 0 or step == steps:
            bar.progress(min(step / steps, 1.0))
            status.text(f"step {step:,} / {steps:,}  ·  fitness {fit:g}  ·  {nb} breakthroughs  ·  "
                        f"{time.time() - t0:.0f} s")
        return False

    if model == "2d":
        ev = rb.evolve2d(steps=steps, fitness=fitness, start=start, seed=seed, n_mutations=n_mutations,
                         k=k, neighborhood=neighborhood, max_steps=max_steps, max_width=max_width,
                         callback=cb)
    else:
        ev = rb.evolve(steps=steps, fitness=fitness, start=start, seed=seed, n_mutations=n_mutations,
                       k=k, max_steps=max_steps, callback=cb)
    bar.empty(); status.empty()
    return ev


def organism_stats(model, rule, max_steps, max_width):
    if model == "2d":
        return rb.stats2d(rule, max_steps=max_steps, max_width=max_width)
    return rb.stats(rule, max_steps=max_steps)


def development_grid_2d(ev, gens=16, ncols=8, max_rows=8, cell=1.1):
    """Early development (generations 0..gens) of each breakthrough organism."""
    bts = ev.breakthroughs[1:] or ev.breakthroughs
    if len(bts) > max_rows:
        idx = np.unique(np.linspace(0, len(bts) - 1, max_rows).round().astype(int))
        bts = [bts[i] for i in idx]
    ts = np.unique(np.linspace(0, gens, ncols).round().astype(int))
    k = ev.rule.k
    vols = [rb.grow2d(r, steps=gens, max_width=ev.params["max_width"]) for _, _, r in bts]
    w = max(v.shape[1] for v in vols)
    fig, axes = plt.subplots(len(vols), len(ts), figsize=(cell * len(ts), cell * len(vols)), squeeze=False)
    for i, (v, (s, f, _)) in enumerate(zip(vols, bts)):
        pad = np.zeros((gens + 1, w, w), dtype=np.uint8)
        o = (w - v.shape[1]) // 2
        pad[:v.shape[0], o:o + v.shape[1], o:o + v.shape[2]] = v[:gens + 1]
        for j, t in enumerate(ts):
            ax = axes[i, j]
            ax.imshow(pad[t], cmap=rb.colormap(k), vmin=0, vmax=k - 1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"step {s}\n{ev.fitness_name} {f:g}", fontsize=7)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# sidebar: model and parameters
# ----------------------------------------------------------------------------

st.sidebar.title("🧬 Ruliobiology")
model = st.sidebar.radio("Model", ["1d", "2d"], format_func=lambda m: {"1d": "1-D (Wolfram's setup)",
                                                                          "2d": "2-D"}[m], horizontal=True)
S = st.session_state
key = lambda name: f"{model}_{name}"          # per-model session-state keys

if model == "2d":
    k = st.sidebar.number_input("Colours k", 2, 6, 4, key="k2d")
    neighborhood = st.sidebar.selectbox("Neighbourhood", ["vonneumann", "moore"],
                                        format_func=lambda n: {"vonneumann": "von Neumann (4)", "moore": "Moore (8)"}[n])
    n_cases = len(ca2d.valid_cases(int(k), len(ca2d.NEIGHBORHOODS[neighborhood])))
    default_steps = 10000
else:
    k = st.sidebar.number_input("Colours k", 2, 8, 5, key="k1d")
    neighborhood = None
    n_cases = len(ca1d.symmetric_cases(int(k)))
    default_steps = 1000
k = int(k)
st.sidebar.caption(f"genome: {n_cases} independent rule cases")

fit_opts = fitness_options(model, k)
fit_name = st.sidebar.selectbox("Fitness", list(fit_opts), key=key("fitness"))
fitness = fit_opts[fit_name]
steps = int(st.sidebar.number_input("Steps of adaptive evolution", 10, 1_000_000, default_steps, step=100, key=key("steps")))
random_seed = st.sidebar.checkbox("Random seed each run", value=False, key=key("rnd"))
seed = int(st.sidebar.number_input("Seed", 0, 2**31 - 1, 1, key=key("seed"), disabled=random_seed))
n_mut = int(st.sidebar.slider("Rule cases mutated per step", 1, 5, 1, key=key("nmut")))
with st.sidebar.expander("Infinite-pattern cutoffs"):
    max_steps = int(st.number_input("Lifetime cap (generations)", 50, 100_000,
                                    rb.DEFAULT_MAX_STEPS if model == "1d" else ca2d.DEFAULT_MAX_STEPS,
                                    step=100, key=key("maxsteps"),
                                    help="A pattern still alive after this many generations counts as infinite (fitness 0)."))
    max_width = None
    if model == "2d":
        max_width = int(st.number_input("Width cap (cells)", 10, 2000, ca2d.DEFAULT_MAX_WIDTH, step=10, key=key("maxwidth"),
                                        help="A pattern wider than this counts as infinite. This is what keeps 2-D runs fast."))

c1, c2 = st.sidebar.columns(2)
run_new = c1.button("Run evolution", type="primary", use_container_width=True)
run_more = c2.button("Continue", use_container_width=True, disabled=key("ev") not in S,
                     help="Keep evolving the current organism for the same number of steps")

with st.sidebar.expander("Load / save by rule number"):
    load_text = st.text_area("Rule number", key=key("loadtext"), height=80,
                             help="Paste a rule number (the genome as one integer) to load that organism")
    if st.button("Load rule", key=key("loadbtn")) and load_text.strip():
        try:
            rule = rule_from_number(model, load_text, k, neighborhood)
            s = organism_stats(model, rule, max_steps, max_width)
            f = fitness(s)
            ev = ca1d.Evolution(rule, f, np.zeros(0), np.zeros(0), [(0, f, rule)], [(0, f, rule)],
                                getattr(fitness, "__name__", "fitness").replace("fit_", ""),
                                dict(steps=0, seed=None, n_mutations=n_mut, k=k, max_steps=max_steps,
                                     max_width=max_width, seed_color=1, fix_background=True,
                                     neighborhood=neighborhood, dimension=2 if model == "2d" else 1))
            S[key("ev")] = ev
            st.success(f"loaded: {ev.fitness_name} {f:g}")
        except Exception as e:  # noqa: BLE001
            st.error(f"could not load: {e}")
    if key("ev") in S:
        st.code(str(S[key("ev")].rule.to_number()), language=None)
        st.caption("current organism's rule number (select and copy)")

st.sidebar.markdown(f"[Wolfram's essay]({WOLFRAM}) · `ruliobiology` {rb.__version__} · "
                    f"{'numba' if rb.HAVE_NUMBA else 'numpy (install numba for speed)'}")

# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------

if run_new or run_more:
    used_seed = int(np.random.randint(2**31 - 1)) if random_seed else seed
    start = S[key("ev")].rule if run_more else None
    if run_more and not random_seed:
        used_seed = seed + S[key("ev")].steps + 1        # continuing with the same seed would repeat the mutations
    with st.spinner("evolving..."):
        ev = run_evolution(model, steps, fitness, start, used_seed, n_mut, k, neighborhood, max_steps, max_width)
    if run_more:
        ev = S[key("ev")].extend(ev)
    ev.params["seed"] = used_seed
    S[key("ev")] = ev
    S.pop(key("viewer"), None)
    st.rerun()                      # re-render with the new organism (and enable "Continue")

# ----------------------------------------------------------------------------
# main area
# ----------------------------------------------------------------------------

if key("ev") not in S:
    st.title("Ruliobiology")
    st.markdown(
        "Adaptive evolution of simple cellular-automaton rules, after Stephen Wolfram's "
        f"[*“Endless Forms Most Beautiful and Most Wonderful”: Ruliobiology and the Theoretical "
        f"Foundations of Biological Form*]({WOLFRAM}) (October 2026).\n\n"
        "Choose a model and parameters in the sidebar and press **Run evolution**. A single lineage starts "
        "from the null rule; each step mutates one rule case at random and keeps the mutant if its fitness "
        "did not decrease. With the default settings a 1-D run takes a few seconds and a 2-D run a minute "
        "or two.")
    st.image(os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "images", "fossil_record_1d.png"),
             caption="A 1-D lineage's fossil record: the organism at each fitness breakthrough.")
    st.stop()

ev = S[key("ev")]
rule = ev.rule
stats = organism_stats(model, rule, ev.params["max_steps"], ev.params.get("max_width"))

st.title(f"{'2-D' if model == '2d' else '1-D'} rulioorganism · {ev.fitness_name} {ev.fitness:g}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("lifetime", stats.lifetime if stats.finite else "∞")
m2.metric("width", stats.width)
m3.metric("cells", f"{stats.cells:,}")
m4.metric("breakthroughs", len(ev.breakthroughs) - 1)
m5.metric("steps", f"{ev.steps:,}")
if ev.steps:
    st.caption(ev.summary() + (f"  ·  seed {ev.params['seed']}" if ev.params.get("seed") is not None else ""))

tab_names = ["Fitness", "Fossil record", "Organism", "Genome", "Development"]
if model == "2d":
    tab_names.append("3-D viewer")
tab_names += ["Gallery", "About"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs["Fitness"]:
    if ev.steps:
        st.markdown("Red dots: the fitness of every mutant tried. Black: the current organism. "
                    "Circles: breakthroughs.")
        show(rb.plot_fitness(ev).figure, "fitness.png")
    else:
        st.info("This organism was loaded from a rule number; run or continue an evolution to see a trajectory.")

with tabs["Fossil record"]:
    st.markdown("The organism at each fitness breakthrough, oldest first.")
    if model == "2d":
        bts = ev.breakthroughs[1:] or ev.breakthroughs
        if len(bts) > 12:
            idx = np.unique(np.linspace(0, len(bts) - 1, 12).round().astype(int))
            bts = [bts[i] for i in idx]
        fig = rb.plot_gallery2d([r for _, _, r in bts], ncols=min(6, len(bts)),
                                titles=[f"step {s}: {ev.fitness_name} {f:g}" for s, f, _ in bts],
                                max_steps=ev.params["max_steps"], max_width=ev.params["max_width"])
        show(fig, "fossil_record.png")
        st.caption("Each organism is shown as its footprint: the colour every cell had when it was last alive.")
    else:
        show(rb.plot_breakthroughs(ev, max_steps=ev.params["max_steps"]), "fossil_record.png")

with tabs["Organism"]:
    if model == "2d":
        vol = rb.grow2d(rule, max_steps=ev.params["max_steps"], max_width=ev.params["max_width"])
        st.markdown(f"Lifetime {vol.shape[0]} generations in a {vol.shape[2]} × {vol.shape[1]} box.")
        n_frames = st.slider("Frames in the filmstrip", 4, 24, 12, key="frames")
        show(rb.plot_filmstrip(vol, n=n_frames, k=k), "filmstrip.png")
        c1, c2, c3 = st.columns(3)
        for col, mode in zip((c1, c2, c3), ("last_color", "birth", "count")):
            with col:
                fig, ax = plt.subplots(figsize=(4, 4))
                rb.plot_projection(vol, mode, ax=ax, k=k)
                show(fig, f"projection_{mode}.png")
    else:
        st.markdown("Time runs downward from the single seed cell at the top.")
        hist = rb.grow(rule, max_steps=ev.params["max_steps"])
        zoom = st.select_slider("Pixels per cell", [1, 2, 3, 4, 6, 8], value=3, key="zoom1d")
        img = np.asarray(rb.colormap(k)(hist))[..., :3]
        img = np.kron(img, np.ones((zoom, zoom, 1)))
        st.image((img * 255).astype(np.uint8), caption=f"{hist.shape[0]} generations × {hist.shape[1]} cells")
        st.download_button("Download organism.png", png(rb.plot_organism(rule, max_steps=ev.params["max_steps"],
                                                                           max_inches=12).figure, dpi=200),
                           file_name="organism.png", mime="image/png")
    st.markdown("Colour fractions: " + ", ".join(f"{COLOR_NAMES[c]} {f:.3f}"
                                                 for c, f in enumerate(stats.color_fractions) if c))

with tabs["Genome"]:
    st.markdown(f"**Rule number** (the genome as one base-{k} integer, {n_cases} digits):")
    st.code(str(rule.to_number()), language=None)
    if model == "2d":
        rows = [{"own colour": COLOR_NAMES[own],
                 **{f"# {COLOR_NAMES[c + 1]} neighbours": n for c, n in enumerate(counts)},
                 "→ becomes": COLOR_NAMES[out]}
                for (own, counts), out in rule.cases() if out != 0 or own != 0]
        st.markdown(f"{len(rows)} of {n_cases} cases do something other than 'stay/turn white':")
        st.dataframe(rows, use_container_width=True, height=400)
    else:
        st.markdown("Each icon: the three-cell neighbourhood above the colour it becomes "
                    f"({n_cases} independent cases; mirror images are implied).")
        show(rb.plot_rule(rule).figure, "rule.png")
        used = rb.used_cases(rule, steps=10)
        st.caption(f"{len(used)} of these cases are used in the first 10 generations of development.")

with tabs["Development"]:
    st.markdown("Early development of each breakthrough organism along the lineage. Wolfram's "
                "observation: the first generations are *conserved* once found, because only a subset "
                "of the genome is expressed early.")
    if len(ev.breakthroughs) < 2:
        st.info("Run an evolution to see a lineage.")
    elif model == "2d":
        gens = st.slider("Generations shown", 4, 40, 16, key="gens2d")
        show(development_grid_2d(ev, gens=gens), "development.png")
    else:
        gens = st.slider("Generations shown", 5, 100, 30, key="gens1d")
        show(rb.plot_development(ev, steps=gens), "development.png")

if model == "2d":
    with tabs["3-D viewer"]:
        st.markdown("The organism's whole life as a space-time volume: x and y are the CA's two "
                    "dimensions, height is time (downward). Drag to rotate, wheel to zoom; shrink the "
                    "spheres and fade a colour to look inside.")
        if key("viewer") not in S:
            vol = rb.grow2d(rule, max_steps=ev.params["max_steps"], max_width=ev.params["max_width"])
            S[key("viewer")] = write_viewer(
                vol, title=f"{ev.fitness_name} {ev.fitness:g}",
                caption=f"{k} colours, {ev.params.get('neighborhood')} neighbourhood, {ev.steps:,} steps "
                        f"of adaptive evolution. Rule number {rule.to_number()}")
        html = S[key("viewer")]
        components.html(html, height=720, scrolling=False)
        st.download_button("Download this viewer as a stand-alone HTML page", html,
                           file_name="organism3d.html", mime="text/html")
        st.caption("The page loads three.js from cdnjs, so it needs an internet connection; rendering uses "
                   "the GPU through WebGL.")

with tabs["Gallery"]:
    st.markdown("Evolve many independent lineages with the sidebar settings and compare them. "
                "Pick one to load it as the current organism.")
    g1, g2, g3 = st.columns([1, 1, 2])
    n_runs = int(g1.number_input("Lineages", 2, 200, 12, key=key("nruns")))
    n_jobs = int(g2.number_input("Processes", 1, max(1, os.cpu_count() or 1), min(4, os.cpu_count() or 1), key=key("njobs")))
    if g3.button("Evolve gallery", key=key("gallerybtn")):
        with st.spinner(f"evolving {n_runs} lineages × {steps:,} steps on {n_jobs} process(es)..."):
            t0 = time.time()
            if model == "2d":
                evs = rb.evolve2d_many(n_runs, steps=steps, seed=seed, n_jobs=n_jobs, fitness=fitness,
                                       n_mutations=n_mut, k=k, neighborhood=neighborhood,
                                       max_steps=max_steps, max_width=max_width)
            else:
                evs = rb.evolve_many(n_runs, steps=steps, seed=seed, n_jobs=n_jobs, fitness=fitness,
                                     n_mutations=n_mut, k=k, max_steps=max_steps)
            for e in evs:
                e.params["seed"] = seed
            S[key("gallery")] = sorted(evs, key=lambda e: -e.fitness)
            st.caption(f"done in {time.time() - t0:.0f} s")
    if key("gallery") in S:
        evs = S[key("gallery")]
        titles = [f"#{i}: {e.fitness_name} {e.fitness:g}" for i, e in enumerate(evs)]
        ncols = min(6, len(evs))
        if model == "2d":
            fig = rb.plot_gallery2d([e.rule for e in evs], ncols=ncols, titles=titles,
                                    max_steps=max_steps, max_width=max_width)
        else:
            fig = rb.plot_gallery([e.rule for e in evs], ncols=ncols, titles=titles, max_steps=max_steps)
        show(fig, "gallery.png")
        pick = st.selectbox("Load lineage", titles, key=key("pick"))
        if st.button("Load it", key=key("pickbtn")):
            S[key("ev")] = evs[titles.index(pick)]
            S.pop(key("viewer"), None)
            st.rerun()
        fits = np.array([e.fitness for e in evs])
        st.caption(f"median {np.median(fits):g}, max {fits.max():g}, "
                   f"{int((fits <= 1).sum())} lineage(s) never got going")

with tabs["About"]:
    st.markdown(f"""
**The model.** A *rulioorganism* is a cellular automaton grown from a single seed cell on an infinite
white background. Its rule is the genotype, the pattern it grows is the phenotype, and its fitness is
(by default) its *lifetime*: the number of generations before every cell is white again. Patterns that
never die (periodic ones, or ones still growing at the cutoffs) have fitness 0. Adaptive evolution is a
single asexual lineage: each step changes one rule case at random and keeps the mutant if its fitness
did not decrease. There is no population, no recombination and no environment.

**1-D** follows Wolfram's essay exactly: symmetric 5-colour nearest-neighbour rules (75 independent cases).
**2-D** uses k-colour outer-totalistic rules on the von Neumann or Moore neighbourhood, which are
symmetric under every rotation and reflection of the square; the organism's form is then a
(time, y, x) volume that the 3-D viewer lets you rotate and look into.

**Reading.** S. Wolfram, [*“Endless Forms Most Beautiful and Most Wonderful”: Ruliobiology and the
Theoretical Foundations of Biological Form*]({WOLFRAM}) (Oct 2026), and the essays it builds on:
[*Why Does Biological Evolution Work?*](https://writings.stephenwolfram.com/2024/05/why-does-biological-evolution-work-a-minimal-model-for-biological-evolution-and-other-adaptive-processes/)
(2024) and [*Foundations of Biological Evolution: More Results & More Surprises*](https://writings.stephenwolfram.com/2024/12/foundations-of-biological-evolution-more-results-more-surprises/) (2024).

The library behind this page is `ruliobiology` (see the README for the API).
""")
