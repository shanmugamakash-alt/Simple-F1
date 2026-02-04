import fastf1 as ff1
import importlib
print('fastf1 version:', getattr(ff1, '__version__', 'unknown'))
print('plotting present in fastf1 dir:', 'plotting' in dir(ff1))
print('plot-related members:', [n for n in dir(ff1) if 'plot' in n.lower()])
try:
    fpl = importlib.import_module('fastf1.plotting')
    print('fastf1.plotting module imported')
    print('plotting public members sample:', [n for n in dir(fpl) if not n.startswith('_')][:40])
except Exception as e:
    print('import fastf1.plotting failed:', repr(e))
