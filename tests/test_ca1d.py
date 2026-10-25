"""pytest suite for ruliobiology.ca1d.  Run with:  pytest -q"""
import numpy as np
import pytest

from ruliobiology import ca1d as rb

ENGINES = ["numpy"] + (["numba"] if rb.HAVE_NUMBA else [])


# ------------------------------------------------------------------ rules ----
def test_symmetric_case_count():
    assert len(rb.symmetric_cases(5)) == 75
    assert len(rb.symmetric_cases(3)) == 18
    for k in (3, 5):
        for i in range(k ** 3):
            assert rb.case_index(*rb.index_case(i, k), k) == i


def test_rules_stay_symmetric_under_mutation():
    rng = np.random.default_rng(0)
    r = rb.Rule.null()
    for _ in range(200):
        r = r.mutated(rng, n_mutations=rng.integers(1, 4))
        assert r.is_symmetric()
        assert r(0, 0, 0) == 0                     # background pinned
    assert r.distance(rb.Rule.null()) > 0
    with pytest.raises(ValueError):
        t = np.zeros(125, np.uint8); t[1] = 2      # (0,0,1) -> 2 without its mirror
        rb.Rule(t)


def test_mutation_changes_exactly_n_cases():
    rng = np.random.default_rng(1)
    r = rb.Rule.random(rng)
    for n in (1, 2, 5):
        m = r.mutated(rng, n_mutations=n)
        assert r.distance(m) == n


def test_rule_number_roundtrip():
    rng = np.random.default_rng(2)
    for k in (3, 5):
        for _ in range(5):
            r = rb.Rule.random(rng, k=k)
            assert rb.Rule.from_number(r.to_number(), k) == r
    assert rb.Rule.null().to_number() == 0
    assert hash(rb.Rule.null()) == hash(rb.Rule.from_number(0))


# ------------------------------------------------------------- phenotype ----
@pytest.mark.parametrize("engine", ENGINES)
def test_null_rule_and_periodic_detection(engine):
    s = rb.stats(rb.Rule.null(), engine=engine)
    assert s == rb.Stats(1, 1, (0, 1, 0, 0, 0))
    # a single cell that keeps itself alive is a fixed point -> infinite
    r = rb.Rule.from_cases({(0, 1, 0): 1})
    assert rb.lifetime(r, engine=engine) == 0
    # 1 -> 2 -> 1 -> ... oscillator -> infinite
    r = rb.Rule.from_cases({(0, 1, 0): 2, (0, 2, 0): 1})
    assert rb.lifetime(r, engine=engine) == 0
    # 1 -> 2 -> 3 -> dies: lifetime 3, and colours counted once each
    r = rb.Rule.from_cases({(0, 1, 0): 2, (0, 2, 0): 3})
    assert rb.stats(r, engine=engine) == rb.Stats(3, 1, (0, 1, 1, 1, 0))
    # grows at light speed forever: hits the max_steps cap
    r = rb.Rule.from_cases({(0, 1, 0): 1, (0, 0, 1): 1, (1, 1, 1): 1})
    assert rb.lifetime(r, engine=engine, max_steps=50) == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_hand_made_triangle(engine):
    # any neighbourhood containing a 1 becomes 1: a solid triangle of width 2t+1
    r = rb.Rule.from_cases({(0, 0, 1): 1, (0, 1, 0): 1, (0, 1, 1): 1,
                            (1, 0, 1): 1, (1, 1, 1): 1})
    h = rb.grow(r, steps=10)
    assert h.shape == (11, 21) and np.all(h[-1] == 1) and h[0, 10] == 1
    s = rb.stats(r, engine=engine, max_steps=30)
    assert s.lifetime == 0 and s.width == 61 and s.counts[1] == 31 ** 2


@pytest.mark.skipif(not rb.HAVE_NUMBA, reason="numba not installed")
def test_numba_and_numpy_engines_agree():
    rng = np.random.default_rng(3)
    r = rb.Rule.null()
    for _ in range(300):
        r = r.mutated(rng)
        a = rb.stats(r, max_steps=400, engine="numba")
        b = rb.stats(r, max_steps=400, engine="numpy")
        assert a == b


def test_grow_matches_stats():
    ev = rb.evolve(steps=150, seed=4, max_steps=500)
    for _, f, r in ev.breakthroughs:
        s = rb.stats(r, max_steps=500)
        h = rb.grow(r, max_steps=500)
        assert s.lifetime == f == h.shape[0]
        assert s.width == h.shape[1]
        assert s.cells == int((h > 0).sum())
        assert np.array_equal(h, h[:, ::-1])          # mirror symmetric
        counts = np.bincount(h.ravel(), minlength=5)
        assert tuple(counts[1:]) == s.counts[1:]


def test_grow_from_patch_and_used_cases():
    ev = rb.evolve(steps=150, seed=5, max_steps=500)
    h = rb.grow(ev.rule, max_steps=500)
    patch = h[h.shape[0] // 2]                         # a row from the middle
    iso = rb.grow(ev.rule, steps=20, initial=patch)
    assert iso.ndim == 2 and iso.shape[0] <= 21
    used = rb.used_cases(ev.rule, steps=10)
    assert used and all(0 < i < 125 for i in used)
    assert rb.used_cases(rb.Rule.null(), steps=1) == {rb.case_index(0, 1, 0),
                                                      rb.case_index(0, 0, 1),
                                                      rb.case_index(1, 0, 0)}


# ------------------------------------------------------------- evolution ----
def test_evolution_is_monotone_and_reproducible():
    a = rb.evolve(steps=300, seed=6, max_steps=500)
    b = rb.evolve(steps=300, seed=6, max_steps=500)
    assert a.rule == b.rule and np.array_equal(a.attempts, b.attempts)
    assert np.all(np.diff(a.best) >= 0)
    fits = [f for _, f, _ in a.breakthroughs]
    assert fits == sorted(fits) and len(set(fits)) == len(fits)
    assert a.breakthroughs[0][1] == 1                  # null rule lives one step
    assert a.fitness == a.best[-1] == rb.lifetime(a.rule, max_steps=500)
    assert all(f > 0 for _, f, _ in a.lineage)         # never accepts an infinite one
    assert "breakthroughs" in a.summary()


def test_other_fitness_functions_and_evolve_many():
    ev = rb.evolve(steps=100, seed=7, fitness=rb.fit_tall, max_steps=300)
    assert ev.fitness_name == "tall" and ev.fitness > 0
    evs = rb.evolve_many(3, steps=50, seed=8, max_steps=300)
    assert len(evs) == 3 and len({e.rule for e in evs}) == 3


def test_k3_rules_work_too():
    ev = rb.evolve(steps=200, seed=9, k=3, max_steps=300)
    assert ev.rule.k == 3 and len(ev.rule.table) == 27 and ev.fitness >= 1


# -------------------------------------------------------------- plotting ----
def test_plots_run_headless():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = rb.evolve(steps=120, seed=10, max_steps=300)
    rb.plot_fitness(ev); rb.plot_breakthroughs(ev); rb.plot_organism(ev.rule)
    rb.plot_rule(ev.rule); rb.plot_development(ev, steps=10)
    rb.plot_gallery([ev.rule, rb.Rule.null()], titles=["a", "b"])
    plt.close("all")
