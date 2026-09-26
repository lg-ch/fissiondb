"""Build the ctypes engine as a platform wheel, without a Python C API shim."""
import os
import platform
from setuptools import Extension,setup
from setuptools.command.build_ext import build_ext


class NativeBuild(build_ext):
    def get_ext_filename(self,name):
        return os.path.join(*name.split('.'))+'.so'


arch=['-march=armv8.2-a+dotprod+fp16'] if platform.machine() in ('aarch64','arm64') else ['-march=x86-64']
setup(ext_modules=[Extension('fissiondb.libfissiondb_anchor',
    sources=['src/anchor.c','src/anchor_live.c'],
    depends=['src/anchor.h','src/anchor_live.h','src/anchor_residual.inc',
             'src/anchor_residual_build.inc','src/anchor_live_pack.inc','src/anchor_live_backup.inc'],
    libraries=['m','uring','roaring','xxhash','pthread','curl'],
    extra_compile_args=['-O3','-std=c11','-fopenmp']+arch,
    extra_link_args=['-fopenmp','-Wl,--no-undefined'])],
    cmdclass={'build_ext':NativeBuild})
