"""Look-ahead en paralelo (4-nov-2026): reparte las M simulaciones de una decision en P procesos con semillas contiguas.
Saudi, vuelta 7, escenario (ii), PPO v10g. Mide el tiempo de reloj del comando completo (incluye cargar el modelo en
cada proceso) y comprueba que la union de los P procesos es identica al barrido de las Figs. 8-10.
Correr con el entorno de results/decision_support_v10gw24belief_20261103/run_20261103.sh."""
import os, shutil, subprocess, sys, time, tempfile
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
WRAP = os.path.join(ROOT, 'scripts', 'gp_configs_fix_20261028.py')
ROLLOUTS = os.path.join(ROOT, 'scripts', 'ds_rollouts_belief_20261012.py')
SWEEP = os.path.join(ROOT, 'results', 'decision_support_v10gw24belief_20261103', 'PPO')
TAG = 'saudi_arabian_nor_real_lap7'
KEYS = ['final_position', 'total_time', 'points', 'n_stops', 'pit_laps', 'masked_hits']
env = dict(os.environ, OMP_NUM_THREADS='1', F1_SIMULATION_DIR=os.path.join(ROOT, 'trained_models', 'simulation_v10eGQ_20261028'),
           F1_RL_AGENTS_DIR=os.path.join(ROOT, 'trained_models', 'retrained_v10g_20261030', 'selected'))
ref = pd.read_csv(os.path.join(SWEEP, TAG + '.csv'))
rows = []
for M, P in [(500, 1), (500, 4), (500, 8), (100, 1), (100, 4), (100, 8)]:
    tmp = tempfile.mkdtemp(dir=sys.argv[1])
    procs, off = [], 0
    t0 = time.time()
    for k in range(P):
        n = M // P + (k < M % P)
        out = os.path.join(tmp, str(k))
        os.makedirs(out)
        shutil.copy(os.path.join(SWEEP, 'initial_compounds.json'), out)
        cmd = [sys.executable, WRAP, ROLLOUTS, '--gp', 'Saudi Arabian Grand Prix', '--driver', 'Driver_NOR',
               '--model-dir', 'Saudi_Arabian_NOR', '--algo', 'PPO', '--M', str(n), '--base-seed', str(1000 + off),
               '--fast-env', '--legal-mask', '--real-2024', '--sim-future', '--following-gap-scale', '0.26',
               '--out-dir', out, '--laps', '7']
        procs.append((subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), off))
        off += n
    rcs = [p.wait() for p, _ in procs]
    wall = time.time() - t0
    parts = []
    for k, (_, o) in enumerate(procs):
        d = pd.read_csv(os.path.join(tmp, str(k), TAG + '.csv'))
        d['rollout'] += o
        parts.append(d)
    u = pd.concat(parts).sort_values(['action', 'rollout']).reset_index(drop=True)
    r = ref[ref.rollout < M].sort_values(['action', 'rollout']).reset_index(drop=True)
    same = len(u) == len(r) and u[KEYS].fillna('').astype(str).equals(r[KEYS].fillna('').astype(str))
    rows.append({'M': M, 'P': P, 'wall_s': round(wall, 1), 'rc_ok': all(c == 0 for c in rcs), 'n_rows': len(u), 'identical_to_sweep': same})
    print(rows[-1], flush=True)
    shutil.rmtree(tmp)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parallel_lookahead_20261104.csv'), index=False)
