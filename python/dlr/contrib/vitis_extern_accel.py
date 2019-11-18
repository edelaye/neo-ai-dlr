"""Registration of fused acceleration operation"""

import os
import tvm
import warnings
import numpy as np

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
def accel_fused(path, kernel_name, input_name, output_name, 
    output_layout, platform, out, *ins):

    #print(kernel_name, input_name, output_name)

    if platform == "DPU":

        # Attach to DPU driver and prepare for running
        n2cube.dpuOpen()
        
        # Create DPU Kernels
        kernel = n2cube.dpuLoadKernel(kernel_name)

        # Create DPU Tasks for kernel
        task = n2cube.dpuCreateTask(kernel, 0)

        # Load image to DPU
        X = ins[0].asnumpy().reshape((-1))
        n2cube.dpuSetInputTensorInHWCFP32(task, input_name, X, len(X))

        # Model run on DPU """
        n2cube.dpuRunTask(task)
        
        # Get the output tensor size 
        size = n2cube.dpuGetOutputTensorSize(task, output_name)
        address = n2cube.dpuGetOutputTensorAddress(task, output_name)

        value = [0 for i in range(size)]

        n2cube.dpuGetTensorData (address, value, size)
        scale = n2cube.dpuGetOutputTensorScale(task, output_name, idx=0)
        value = np.array(value).astype(np.float32)/scale

        # DPU output is in NHWC
        if output_layout == 'NCHW':
            value = np.transpose(value,(0,3,1,2))

        
    elif platform == 'XDNN':
        # CREATE A HANDLE FOR FPGA COMMUNICATION
        platform = os.environ.get('MLSUITE_PLATFORM')#"alveo-u200"
        xclbin = "/workspace/MLsuite/overlaybins/" + platform.lower() + "/overlay_4.xclbin"


        layout = output_layout 
        
        args_dict = {
            'xclbin'     : xclbin,
            'netcfg'     : str(path  + "/_compiler.json" ),
            'quantizecfg': str(path  + "/_quantizer.json"),
            'weights'    : str(path  + "/_weights.h5"    ), 
            'scaleA'     : 1,
            'scaleB'     : 1,
            'PE'         : 0,
            'batch_sz'   : ins[0].shape[0],
            'inshape'    : tuple(ins[0].shape[1:])
        }

          
        ret, handles = xdnn.createHandle(xclbin)
     
        if ret != 0:
            raise ValueError("ERROR: Unable to create handle to FPGA")
        
        args = xdnn_io.make_dict_args(args_dict)
     
        fpgaRT = xdnn.XDNNFPGAOp(handles,args)
     
        fpgaInput = fpgaRT.getInputs()
        fpgaOutput = fpgaRT.getOutputs()
     
        batch_array = np.empty(((ins[0].shape[0],) + tuple(ins[0].shape[1:])), dtype=np.float32, order='C')
        data_paths = [ins[0].asnumpy()]
     
        for i in range(0, len(data_paths), ins[0].shape[0]):
            for j, d in enumerate(data_paths[i:i + ins[0].shape[0]]):
                batch_array[j, ...] = d
        
        fpgaInput[list(fpgaInput.keys())[0]] = batch_array
     
        # Execute xdnn
        fpgaRT.execute(fpgaInput, fpgaOutput)

        key, value  = fpgaOutput.popitem()
     
        # xDNN output is in NCHW format
        if str(layout) == 'NHWC':
            value = np.transpose(value,(0,2,3,1))
    else:
        raise ValueError("Unknown platform: {}".format(platform))
        
    tvm.nd.array(value).copyto(out)
