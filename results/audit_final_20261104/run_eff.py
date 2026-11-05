import sys, os, types
R=os.environ['RROOT']; sys.path[:0]=[R, os.path.join(R,'scripts')]
src=open(os.path.join(R,'scripts','decision_support_rollouts.py')).read()
old="""        if act != 0:
            n_stops += 1
            pit_laps.append(int(env.lap_number))
        obs, _, done, _, info = env.step(act)
"""
assert src.count(old)==1
new="""        _lap=int(env.lap_number); _c=env.agent_driver.compound; _t=_tlget(env)
        obs, _, done, _, info = env.step(act)
        if act != 0 and (env.agent_driver.compound!=_c or _tlget(env)<=_t and _tlget(env)!=_t):
            n_stops += 1
            pit_laps.append(_lap)
"""
src=src.replace(old,new)
src="def _tlget(env):\n    a=env.agent_driver\n    return getattr(a,'tyre_life',getattr(a,'tire_life',getattr(a,'tyrelife',None)))\n"+src
m=types.ModuleType('decision_support_rollouts'); m.__file__=os.path.join(R,'scripts','decision_support_rollouts.py'); sys.modules['decision_support_rollouts']=m
exec(compile(src,m.__file__,'exec'),m.__dict__)
import runpy
sys.argv=[os.path.join(R,'scripts','ds_rollouts_belief_20261012.py')]+sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
