from .autograd import *
import numpy as np

def tensor(data, device="cpu", requires_grad=True):
    return AutogradTensor(data=data, device=device, requires_grad=requires_grad)

def parameter(data, device="cpu", requires_grad=True):
    return Parameter(data=data, device=device, requires_grad=requires_grad)

def zeros(shape, device="cpu", requires_grad=True):
    return AutogradTensor(np.zeros(shape), device=device, requires_grad=requires_grad)

def ones(shape, device="cpu", requires_grad=True):
    return tensor(np.ones(shape), requires_grad)