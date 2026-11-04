"""2026-11-03 — Tiempos de entrenamiento desde sacct (sacct_20261103.txt: trabajo|tarea|estado|segundos|CPU). Une cada tarea con su
linea del TSV. Multi-piloto v10g: PPO 18138645, otros 18148535. Mono-piloto v10eGQ: 18037137 y, para DQN, 18037186 y 18037284.
Las horas son de reloj en el cluster NLHPC, AMD EPYC 9224, con 4 CPU por tarea multi-piloto. Salida: training_times_20261103.csv"""
import os, re
import pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(H)); C = f'{R}/reproduce/cluster'
T = {'18138645': 'tasks_v10g_ppo8_20261030.tsv', '18148535': 'tasks_v10g_others_rppo05_20261031.tsv',
     '18037137': 'tasks_onedriver_v10eGQ_20261028.tsv', '18037186': 'tasks_onedriver_v10eGQ_20261028.tsv', '18037284': 'tasks_onedriver_v10eGQ_20261028.tsv'}
d = pd.read_csv(f'{H}/sacct_20261103.txt', sep='|', names=['job', 'id', 'state', 'sec', 'cpu'], dtype={'job': str})
print(d.groupby(['job', 'state']).size())
d = d[d.state == 'COMPLETED'].copy(); d = d[d.id.str.contains('_')]
d['i'] = d.id.str.split('_').str[1]; d = d[d.i.str.isdigit()]; d['i'] = d.i.astype(int)
rows = []
for j, f in T.items():
    names = [l.split('\t')[0] for l in open(f'{C}/{f}') if l.strip()]
    z = d[d.job == j].copy(); z['name'] = [names[i - 1] for i in z.i]; rows.append(z)
d = pd.concat(rows); d['algo'] = d.name.str.split('__').str[-1]; d['gp'] = d.name.str.split('__').str[-2]; d['seed'] = d.name.str.split('__').str[1]
d['setting'] = d.name.str.startswith('onedriver').map({True: 'single', False: 'multi'})
# la ultima corrida de cada tarea (DQN mono-piloto se relanzo)
d = d.sort_values('job').drop_duplicates(['setting', 'gp', 'seed', 'algo'], keep='last')
g = d.groupby(['setting', 'algo']).sec.agg(['size', 'median', 'min', 'max', 'sum', 'mean']); g[['median', 'min', 'max', 'mean']] = (g[['median', 'min', 'max', 'mean']] / 3600).round(2); g['total_h'] = (g['sum'] / 3600).round(0)
print(g.drop(columns='sum')); print(d.groupby('setting').cpu.agg(['min', 'max']))
print('por GP, multi PPO (h):', (d[(d.setting == 'multi') & (d.algo == 'PPO')].groupby('gp').sec.median() / 3600).round(2).to_dict())
g.to_csv(f'{H}/training_times_20261103.csv')
