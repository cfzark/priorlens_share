"""Build the optional model-specific C++ engine for the active Python ABI."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import pybind11

ROOT = Path(__file__).resolve().parents[1]
dest = ROOT/'function/prioreffect'/('_lupus_fast'+sysconfig.get_config_var('EXT_SUFFIX'))
command = shlex.split(os.environ.get('CXX','c++')) + ['-O3','-shared','-std=c++17','-fPIC',
    '-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],
    str(ROOT/'cpp/lupus_fast.cpp'),'-o',str(dest)]
if sys.platform == 'darwin':
    command += ['-undefined','dynamic_lookup']
subprocess.run(command,check=True)
print('Built',dest.relative_to(ROOT))
