import numpy as cpu_np
import cupy as gpu_np

def numpy_choose_cpu_gpu(choose = 'cpu'):

    if choose == 'cpu':

        return cpu_np

    elif choose == 'cuda':

        return gpu_np
