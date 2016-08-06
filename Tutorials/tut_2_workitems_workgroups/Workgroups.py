import pyopencl as cl
import numpy as np


def setContext_GPUdevice(platforms):
    context = None
    for platform in platforms:
        try:
            if context == None:
                context = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platform)])
        except:
            continue
    return context


platforms = cl.get_platforms()
context = setContext_GPUdevice(platforms)


end = 1