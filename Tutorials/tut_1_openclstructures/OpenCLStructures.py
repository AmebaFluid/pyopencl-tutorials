import pyopencl as cl

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
contextB = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])
contextC = cl.Context(dev_type=cl.device_type.CPU, properties=[(cl.context_properties.PLATFORM, platforms[0])])

# OpenCL tracks how often a context structure is accessed. This number is called the reference count.
r_count_A = contextA.get_info(cl.context_info.REFERENCE_COUNT)


###########
# Program #
###########

###################
# Kernel Function #
###################

#################
# Command Queue #
#################