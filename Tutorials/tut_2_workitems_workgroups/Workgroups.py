import pyopencl as cl
import numpy as np


def setContext_GPUdevice(platforms):
    context = None
    for platform in platforms:
        try:
            if context == None:
                context = cl.Context(dev_type=cl.device_type.GPU, properties=[(cl.context_properties.PLATFORM, platform)])
            else:
                context_devices = context.get_info(cl.context_info.DEVICES)
        except:
            continue
    return context


platforms = cl.get_platforms()
context = setContext_GPUdevice(platforms)


end = 1