import pyopencl as cl
import numpy as np

## Tutorial 1: OpenCL structures
# In this part you can learn, how to build up the basic structures that are needed for parallel programming
# in OpenCL/PyOpenCL. This code only contains the absolute minimum you need for an application. If you are interested in
# the structures in more detail, set breakpoints and inspect the object attributes, visit the documentation of PyOpenCL
# (https://documen.tician.de/pyopencl/) or continue with the next tutorials.


############
# Platform #
############

# Looking for different openCL implementations on your hardware (platforms):
platforms = cl.get_platforms()


##########
# Device #
##########

# Looking for available devices (CPU, GPU, Accelerator) for each platform:
platform_devices = []
for platform in platforms:
    devices = platform.get_devices()
    platform_devices.append(devices)


###########
# Context #
###########

# A context is a structure that manages a set of devices of one specific platform.
# A host application is able to work with contexts of different platforms.
# Devices of contexts from different platforms are not able to share resources.

# There are two ways to create a context:
# 1. for specific devices. In this case, create a context for all devices of the first platform:
contextA = cl.Context(platform_devices[0])
# 2. for all devices for the first platform of a given type:
try:
    contextB = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
except:
    contextC = cl.Context(dev_type=cl.device_type.CPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])

# OpenCL tracks how often a context structure is accessed. This number is called the reference count.
r_count_A = contextA.get_info(cl.context_info.REFERENCE_COUNT)


###################
# Kernel Function #
###################

# A kernel function represents a function that should be executed in parallel.
# For example have a look at this function:

def subtract_offset(vector, offset):
    shape = vector.shape
    if len(shape) == 1:
        result = np.zeros(len(vector))
        for element in range(0, len(vector)):
            result[element] = vector[element] - offset
        return result
    else:
        print("The subtract_offset method works only for one-dimensional arrays.")

kernelstringA = """__kernel void exampleKernelFunction(global int* inputdata, global int* outputdata){

     int loopindex = get_global_id(0);
     int offset = 1;
     outputdata[loopindex] = inputdata[loopindex] - offset;

}"""

filename = 'exampleKernelFunction.cl'
file = open(filename, 'r')
kernelstringB = "".join(file.readlines())


###########
# Program #
###########

# The kernel function is build by an OpenCL-program, that can contain multiple kernels.

programA = cl.Program(contextA, kernelstringA).build()
programB = cl.Program(contextA, kernelstringB).build()

# Every time you use program.kernel_name a new kernel object is produced.




#################
# Command Queue #
#################


#########################
# Kernel Execution Time #
#########################


###############
# Simple Test #
###############

testvector = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int32)
offset = 1
cpuresult = subtract_offset(testvector, offset)