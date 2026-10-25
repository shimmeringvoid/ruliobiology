"""pytest suite for ruliobiology.ca2d and ruliobiology.viewer3d.  Run with:  pytest -q"""
import base64

import numpy as np
import pytest

from ruliobiology import ca1d as rb
from ruliobiology import ca2d as rb2
from ruliobiology import viewer3d

ENGINES = ["numpy"] + (["numba"] if rb.HAVE_NUMBA else [])


def life_reference(grid, generations):
    """Game of Life on a big torus, cropped to the live cells (for checking)."""
    from scipy.signal import convolve2d
    g = np.zeros((160, 160), dtype=np.int64)
    g[70:70 + grid.shape[0], 70:70 + grid.shape[1]] = grid
    ker = np.ones((3, 3)); ker[1, 1] = 0
    for _ in range(generations):
        n = convolve2d(g, ker, mode="same", boundary="wrap")
        g = ((n == 3) | ((g == 1) & (n == 2))).astype(np.int64)
    return rb2._trim(g.astype(np.uint8))[2]


# ------------------------------------------------------------------ rules ----
def test_case_counts_and_index_roundtrip():
    assert [len(rb2.valid_cases(k, 8)) for k in (2, 3, 4, 5)] == [18, 135, 660, 2475]
    assert [len(rb2.valid_cases(k, 4)) for k in (2, 3, 4, 5)] == [10, 45, 140, 350]
    for k in (3, 4):
        for own, counts in rb2.valid_cases(k, 8):
            assert rb2.index_case(rb2.case_index(own, counts, k), k) == (own, counts)


def test_mutation_and_rule_number_roundtrip():
    rng = np.random.default_rng(0)
    for nb in ("moore", "vonneumann"):
        r = rb2.Rule2D.random(rng, k=4, neighborhood=nb)
        assert r(0, (0, 0, 0)) == 0
        m = r.mutated(rng, n_mutations=3)
        assert r.distance(m) == 3 and m.neighborhood == nb and m(0, (0, 0, 0)) == 0
        assert rb2.Rule2D.from_number(r.to_number(), 4, nb) == r
        assert r != rb2.Rule2D.null(4, nb) and hash(r) != hash(m)
    with pytest.raises(ValueError):
        rb2.Rule2D.from_cases({(0, (5, 0, 0)): 1}, k=4, neighborhood="vonneumann")


# --------------------------------------------------------------- dynamics ----
def test_game_of_life_from_seed_and_from_pattern():
    life = rb2.Rule2D.life_like()
    assert rb2.stats2d(life, engine="numpy") == rb2.Stats2D(1, 1, 1, (0, 1))
    rpent = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    for gens in (10, 40):
        vol = rb2.grow2d(life, steps=gens, initial=rpent)
        assert vol.shape[0] == gens + 1
        mine = rb2._trim(vol[-1])[2]
        assert np.array_equal(mine, life_reference(rpent, gens))
    # a block is a still life -> periodic -> infinite
    block = np.ones((2, 2), dtype=np.uint8)
    vol = rb2.grow2d(life, steps=5, initial=block)
    assert vol.shape == (6, 2, 2) and vol.all()


@pytest.mark.parametrize("engine", ENGINES)
def test_simple_rules(engine):
    # seed turns colour 2, then dies: lifetime 2
    r = rb2.Rule2D.from_cases({(1, (0, 0, 0)): 2}, k=4)
    assert rb2.stats2d(r, engine=engine) == rb2.Stats2D(2, 1, 1, (0, 1, 1, 0))
    # seed persists forever: periodic
    r = rb2.Rule2D.from_cases({(1, (0, 0, 0)): 1}, k=4)
    assert rb2.lifetime2d(r, engine=engine) == 0
    # light-speed diamond growth hits the width cap -> infinite
    r = rb2.Rule2D.from_cases({(1, (0, 0, 0)): 1, (0, (1, 0, 0)): 1, (0, (2, 0, 0)): 1,
                               (1, (1, 0, 0)): 1, (1, (2, 0, 0)): 1, (1, (3, 0, 0)): 1,
                               (1, (4, 0, 0)): 1}, k=4)
    s = rb2.stats2d(r, engine=engine, max_width=21)
    assert s.lifetime == 0 and s.width == 23
    vol = rb2.grow2d(r, steps=3)
    assert vol.shape == (4, 7, 7) and int((vol[-1] > 0).sum()) == 25   # diamond of radius 3


@pytest.mark.skipif(not rb.HAVE_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("nb", ["moore", "vonneumann"])
def test_engines_agree(nb):
    ev = rb2.evolve2d(steps=1500, seed=2, k=4, neighborhood=nb, max_steps=100, max_width=40)
    for _, f, r in ev.lineage[-40:]:
        a = rb2.stats2d(r, max_steps=100, max_width=40, engine="numba")
        b = rb2.stats2d(r, max_steps=100, max_width=40, engine="numpy")
        assert a == b


def test_grow_matches_stats_and_symmetry():
    ev = rb2.evolve2d(steps=3000, seed=3, max_steps=200, max_width=60)
    assert ev.fitness > 1 and ev.params["dimension"] == 2
    for _, f, r in ev.breakthroughs[-3:]:
        s = rb2.stats2d(r, max_steps=200, max_width=60)
        vol = rb2.grow2d(r, max_steps=200, max_width=60)
        assert s.lifetime == f == vol.shape[0]
        assert s.width == max(vol.shape[1:])
        assert s.cells == int((vol > 0).sum())
        assert s.max_area == max(int((g > 0).sum()) for g in vol)
        for g in vol:                                   # D4 symmetry
            assert np.array_equal(g, g[::-1]) and np.array_equal(g, g.T)
        assert tuple(np.bincount(vol.ravel(), minlength=4)[1:]) == s.counts[1:]


def test_projections():
    vol = np.zeros((3, 2, 2), dtype=np.uint8)
    vol[0, 0, 0] = 1; vol[1, 0, 0] = 2; vol[2, 1, 1] = 3
    assert rb2.projection(vol, "last_color").tolist() == [[2, 0], [0, 3]]
    assert rb2.projection(vol, "birth").tolist() == [[1, 0], [0, 3]]
    assert rb2.projection(vol, "death").tolist() == [[2, 0], [0, 3]]
    assert rb2.projection(vol, "count").tolist() == [[2, 0], [0, 1]]


# ----------------------------------------------------------------- viewer ----
def test_rle_roundtrip_and_viewer_html(tmp_path):
    vol = np.random.default_rng(0).integers(0, 4, size=(6, 9, 300)).astype(np.uint8)
    vol[:, :, 20:280] = 0
    raw = base64.b64decode(viewer3d.rle_encode(vol))
    out = np.zeros(vol.size, np.uint8); pos = 0
    for i in range(0, len(raw), 3):
        n = raw[i + 1] | (raw[i + 2] << 8)
        out[pos:pos + n] = raw[i]; pos += n
    assert pos == vol.size and np.array_equal(out.reshape(vol.shape), vol)
    path = tmp_path / "v.html"
    html = viewer3d.write_viewer(vol, str(path), title="Test organism", caption="c")
    assert path.exists() and html.startswith("<!doctype html>")
    assert "<title>Test organism</title>" in html and viewer3d.THREE_JS in html
    frag = viewer3d.write_viewer(vol, standalone=False)
    assert frag.startswith("<title>") and "<html" not in frag


def test_plots_run_headless():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = rb2.evolve2d(steps=1000, seed=4, max_steps=100, max_width=40)
    vol = rb2.grow2d(ev.rule, max_steps=100, max_width=40)
    rb2.plot_filmstrip(vol); rb2.plot_projection(vol); rb2.plot_projection(vol, "birth")
    rb2.plot_gallery2d([ev.rule, rb2.Rule2D.null()], max_steps=100, max_width=40)
    rb.plot_fitness(ev)
    plt.close("all")
