"""Registration of fused acceleration operation"""


import tvm
import numpy as np

from ctypes import *
import ctypes
import os
import warnings

# TEMP
try:
    import xfdnn.rt.xdnn as xdnn
    import xfdnn.rt.xdnn_io as xdnn_io
    from xfdnn.rt import xdnn, xdnn_io
except:
    warnings.warn("Could not import xfdnn libraries")

try:
    from dnndk import n2cube, dputils
except:
    warnings.warn("Could not import dnndk n2cube")

@tvm.register_func("tvm.accel.accel_fused")
def accel_fused(graph_path, output_layout, model_name, platform, out, *ins):
    raise NotImplementedError("")