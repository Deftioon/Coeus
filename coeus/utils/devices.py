import numpy as np
import cupy as cp

class device:
    def __init__(self, device):
        self.device = device
        if device == "cpu":
            self.module = np
        elif device == "gpu":
            self.module = cp
        else:
            raise ValueError("Unsupported device. Use 'cpu' or 'gpu'.")