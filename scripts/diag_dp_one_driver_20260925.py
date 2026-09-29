"""Exact DP over the single-driver env semantics (f1_env_one_driver.py) vs QMIP.

Env: lap 0 picks compound; laps 1..n pay intercept+eff[c]+deg[c]*tl+pit_in*pin+pit_out*pout_prev;
pit -> tl=0, compound=new, pit_out next lap. Legal = at least one pit to a different compound.
Option --no-last-lap-pit forbids pitting on the final lap (it only pays pit_in).
"""
import sys, os, pickle, functools
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config, compat; compat.setup_all()
from plot_fig5_stints_one_driver import SELECTED_GP, LAPS_DD
from optimization.utils_qmip import extract_simplified_calculator

sys.setrecursionlimit(100000)


def solve(calc, comps, n, max_stops=3, allow_last=True):
    @functools.lru_cache(maxsize=None)
    def f(lap, c, tl, pout, changed, stops, start):
        # cost of laps lap..n given state at start of `lap`
        if lap > n:
            return (0.0, ()) if changed else (float('inf'), ())
        best = (float('inf'), ())
        # no pit
        t = calc.predict(c, tl, 0.0, pout)
        r = f(lap + 1, c, tl + 1, 0.0, changed, stops, start)
        best = min(best, (t + r[0], r[1]))
        if stops < max_stops and (allow_last or lap < n):
            for nc in comps:
                t = calc.predict(c, tl, 1.0, pout)
                r = f(lap + 1, nc, 0, 1.0, changed or nc != c, stops + 1, start)
                best = min(best, (t + r[0], ((lap, nc),) + r[1]))
        return best
    res = []
    for c0 in comps:
        t, plan = f(1, c0, 0, 0.0, False, 0, c0)
        res.append((t, c0, plan))
    f.cache_clear()
    return min(res)


def fmt(c0, plan, n):
    s, prev, cur = [], 1, c0
    for lap, nc in plan:
        s.append(f"C{cur}({lap - prev + 1})"); prev, cur = lap + 1, nc
    s.append(f"C{cur}({n - prev + 1})")
    return ' '.join(s)


def load(dirpath, gp):
    return pickle.load(open(os.path.join(dirpath, f'{gp}_fast_predictor.pkl'), 'rb'))


if __name__ == '__main__':
    dirs = {'sim': str(config.SIMULATION_DIR)}
    for gp, comps_s in SELECTED_GP.items():
        comps = [int(x[1:]) for x in comps_s]; n = LAPS_DD[gp]
        print(f"\n{gp} ({n} laps)")
        for tag, d in dirs.items():
            try:
                calc = extract_simplified_calculator(load(d, gp), comps)
            except Exception as e:
                print(f"  {tag}: ERROR {e}"); continue
            t1, c1, p1 = solve(calc, comps, n, allow_last=False)
            t2, c2, p2 = solve(calc, comps, n, allow_last=True)
            print(f"  {tag:5s} DP (sin pit última vuelta): {t1:9.2f}  {fmt(c1, p1, n)}")
            if t2 < t1 - 1e-6:
                print(f"  {tag:5s} DP (con pit última vuelta): {t2:9.2f}  {fmt(c2, p2, n)}   <-- loophole")
            print(f"        eff={ {k: round(v, 3) for k, v in calc.compound_effects.items()} } "
                  f"deg={ {k: round(v, 4) for k, v in calc.tyre_degradation.items()} } "
                  f"pit={calc.pit_in_cost:.2f}+{calc.pit_out_cost:.2f} int={calc.intercept:.2f}")
