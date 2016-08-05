import pyopencl as cl
import numpy as np

## Tutorial 1: OpenCL structures
# In this part you can learn, how to build up the basic structures that are needed for parallel programming
# in OpenCL/PyOpenCL. This code only contains the absolute minimum you need for an application. If you are interested in
# the structures in more detail, set breakpoints and inspect the object attributes, visit the documentation of PyOpenCL
# (https://documen.tician.de/pyopencl/) or continue with the next tutorials.


#################
# Tutorial data #
#################
testvector = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(np.int32)
outputvector = np.zeros(testvector.shape, dtype=np.int32)
offset = 1

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
    contextB = cl.Context(dev_type=cl.device_type.CPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])

# OpenCL tracks how often a context structure is accessed. This number is called the reference count.
r_count_A = contextA.get_info(cl.context_info.REFERENCE_COUNT)


# -> The tutorial continues with contextA and the devices of the first platform.

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

# The inner execution of the for-loop, in this case the subtraction of an offset value, is called a work-item.
# One individual execution inside the loop corresponds to one work-item.
# In contrast the kernel is a collection of execution tasks, which often contains multiple work-items.
# One individual execution is defined in the kernel function:

kernelstringA = """__kernel void exampleKernelFunction(global int* inputdata, global int* outputdata){

     int loopindex = get_global_id(0);
     int offset = 1;
     outputdata[loopindex] = inputdata[loopindex] - offset;

}"""

# The global_id(0) matches the loop-index. If nested loops should be processed in parallel, global_id(0) corresponds
# to the first loop-index whereas global_id(1) corresponds to the second and so forth.

# The kernel function can be written to an external file including the .cl ending:
filename = 'exampleKernelFunction.cl'
file = open(filename, 'r')
kernelstringB = "".join(file.readlines())


###########
# Program #
###########

# The kernel function is build by an OpenCL-program, that can contain multiple kernels.
# In this tutorial a program is used only for one kernel function.
# Both ways of kernel-definition (inline or external file) are build by a program structure:

programA = cl.Program(contextA, kernelstringA).build()
programB = cl.Program(contextA, kernelstringB).build()


#################
# Command Queue #
#################

queueA = cl.CommandQueue(contextA, device=contextA.devices[0], properties=cl.command_queue_properties.PROFILING_ENABLE)


##################
# Memory Objects #
##################

inputbuffer = cl.Buffer(contextA, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=testvector)
outputbuffer = cl.Buffer(contextA, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=outputvector)


#########################
# Kernel Execution Time #
#########################

event1 = programA.exampleKernelFunction(queueA, testvector.shape, None, inputbuffer, outputbuffer)
events = [event1]
cl.enqueue_copy(queueA, outputvector, outputbuffer, wait_for=events)

event1_enqueued = event1.profile.QUEUED
event1_submitted = event1.profile.SUBMIT
event1_execution_start = event1.profile.START
event1_execution_end = event1.profile.END
event1_execution_time = event1_execution_end - event1_execution_start

# Every time you use program.kernel_name a new kernel object is produced.
example_kernel = programA.exampleKernelFunction
example_kernel.set_args(inputbuffer, outputbuffer)
event2 = cl.enqueue_nd_range_kernel(queueA, example_kernel, testvector.shape, None)

###############
# Simple Test #
###############
result = subtract_offset(testvector, offset)
result_opencl = outputvector

end = 1