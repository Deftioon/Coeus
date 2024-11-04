from . import autograd

def tensor(data, device="cpu", requires_grad=True):
    return autograd.tensor(data, device=device, requires_grad=requires_grad)

def parameter(data, device="cpu"):
    return autograd.tensor(data, device=device)

def inverse(tensor):
    return tensor.inverse()

def transpose(tensor):
    return tensor.transpose()

def sum(tensor, axis=None):
    return tensor.sum(axis=axis)

def mean(tensor, axis=None):
    return tensor.mean(axis=axis)

def hstack(*tensors):
    return tensor(tensors[0].module.hstack([i.data for i in tensors]))

def vstack(*tensors):
    return tensor(tensors[0].module.vstack([i.data for i in tensors]))

def arange(*args):
    if len(args) != 3:
        raise ValueError("arange() takes 3 arguments (start, stop, step)")
    
    list_obj = [i for i in range(args[0], args[1], args[2])]
    return tensor(list_obj)

def ones(shape, device="cpu"):
    return tensor(autograd.ones(shape), device=device)

def zeros(shape, device="cpu"):
    return tensor(autograd.zeros(shape), device=device)