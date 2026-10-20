"""2026-10-14 — Miami en simulation_v6_20261014: probabilidades por carrera de SC y VSC = frecuencia empirica
2022-23 (Miami se corre desde 2022; SC y VSC en 2022, ninguno en 2023) = 0.50 / 0.50, mismo criterio que
Aguad y Thraves (EJOR) para los demas GP. Antes: 0.10 / 0.15, de origen no documentado.
El config.json del arbol v6 era un link al arbol publicado (trained_models/simulation, que no se toca): se
reemplaza el link por un archivo propio. Correr desde la raiz del repo; idempotente.
"""
import os, sys, json
sys.path.insert(0, '.')
import config
gdir = config.MODELS_DIR / 'simulation_v6_20261014' / 'Miami Grand Prix'
p = gdir / 'config.json'
with open(p) as fh:
    cfg = json.load(fh)
print('antes:', cfg['sc_prob'], cfg['vsc_prob'])
cfg['sc_prob'], cfg['vsc_prob'] = 0.5, 0.5
if os.path.islink(p):
    os.unlink(p)
with open(p, 'w') as fh:
    json.dump(cfg, fh, indent=2)
readme = gdir.parent / 'README.md'
note = '\n2026-10-14 (scripts/patch_miami_yf_v6_20261014.py): Miami sc_prob = vsc_prob = 0.50 (frecuencia 2022-23); config.json propio.\n'
if note not in open(readme).read():
    open(readme, 'a').write(note)
print('despues:', cfg['sc_prob'], cfg['vsc_prob'], '->', p)
