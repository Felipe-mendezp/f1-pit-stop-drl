import sys,os; ROOT=os.getcwd(); sys.path[:0]=[ROOT,'src',os.path.join(ROOT,'results','sim_v10eGQ_20261028'),os.path.join(ROOT,'scripts')]
os.environ['F1_TYRELIFE_2021']='v10eGQ'
import pandas as pd, numpy as np
import config, compat; compat.setup_all()
sys.path.append(os.path.join(ROOT,'results','tyre_coef_study_20260930'))
import importlib.util
import models.regressions as R
import common_v10_20261027 as C
C.patch_regressions(R)
sp=importlib.util.spec_from_file_location('P',os.path.join(ROOT,'results','tyre_coef_study_20260930','pooled_20261004.py')); P=importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
d=P.hist_laps_A(); print('hist ER rows',(d.GP.str.contains('Emilia')).sum(),'Dutch 2023',len(d[(d.GP=='Dutch Grand Prix')&(d.Year==2023)]),'Monaco',d[d.GP.str.contains('Monaco')].Year.unique(), 'n editions',d.groupby(['Year','GP']).ngroups)
