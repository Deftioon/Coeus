from .devices import device
import numpy as np

class tensor(device):
    def __init__(self, data=None, requires_grad=True, operation=None, device="cpu"):
        super().__init__(device)
        self.data = self.module.array(data)
        self.requires_grad = requires_grad
        self.operation = operation
        self.children = []
        self.shape = self.data.shape

        if self.requires_grad:
            self.grad = self.module.zeros_like(self.data)

    def __repr__(self):
        return f"Coeus: Tensor[{str(self.shape)[1:-1]}](\n{self.data},\nrequires_grad={self.requires_grad})"
    
    def backward(self, grad=None):
        if not self.requires_grad:
            return ValueError("Tensor does not support gradients.")

        if grad is None:
            grad = self.module.ones_like(self.data)
        
        self.grad += grad.astype(self.grad.dtype)

        if self.operation is not None:
            self.operation.backward(grad)
    
    def zero_grad(self):
        self.grad = self.module.zeros_like(self.grad)

    def zero_all(self):
        self.grad = self.module.zeros_like(self.grad)
        for child in self.children:
            child.zero_all()
    
    def to(self, device):
        self.data = self.data.to(device)
        self.grad = self.grad.to(device)
        self.device = device
    
    def __add__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return add(self, other, self.module).z
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return mul(self, other, self.module).z
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return sub(self, other, self.module).z
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return div(self, other, self.module).z
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __neg__(self):
        return neg(self, self.module).z
    
    def __pow__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return pow(self, other, self.module).z
    
    def __rpow__(self, other):
        return self.__pow__(other)
    
    def __matmul__(self, other):
        other = other if isinstance(other, tensor) else tensor(other, device=self.device)
        return matmul(self, other, self.module).z
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)
    
    def T(self):
        return transpose(self, self.module).z
    
    def inverse(self):
        return inverse(self, self.module).z
    
    def exp(self):
        return exp(self, self.module).z
    
    def log(self):
        return log(self, self.module).z

    def sum(self, axis=None):
        return tensor(self.data.sum(axis=axis))
    
    def mean(self, axis=None):
        return tensor(self.data.mean(axis=axis))
    
class add:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data + y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        self.x.backward(grad)
        self.y.backward(grad)

class mul:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data * y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        self.x.backward(grad * self.y.data)
        self.y.backward(grad * self.x.data)

class div:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data / y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        self.x.backward(grad / self.y.data)
        self.y.backward(-grad * self.x.data / (self.y.data ** 2))

class sub:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data - y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        self.x.backward(grad)
        self.y.backward(-grad)

class neg:
    def __init__(self, x, module):
        self.module = module
        self.x = x
        self.operation = self

        self.z = tensor(-x.data, operation=self)

        x.children.append(self)

    def backward(self, grad):
        self.x.backward(-grad)

class pow:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data ** y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        self.x.backward(grad * self.y.data * (self.x.data ** (self.y.data - 1)))
        self.y.backward(grad * (self.x.data ** self.y.data) * self.module.log(self.x.data))

class exp:
    def __init__(self, x, module):
        self.module = module
        self.x = x
        self.operation = self

        self.z = tensor(self.module.exp(x.data), operation=self)

        x.children.append(self)

    def backward(self, grad):
        self.x.backward(grad * self.z.data)

class log:
    def __init__(self, x, module):
        self.module = module
        self.x = x
        self.operation = self

        self.z = tensor(self.module.log(x.data), operation=self)

        x.children.append(self)

    def backward(self, grad):
        self.x.backward(grad / self.x.data)

class matmul:
    def __init__(self, x, y, module):
        self.module = module
        self.x = x
        self.y = y
        self.operation = self

        self.z = tensor(x.data @ y.data, operation=self)

        x.children.append(self)
        y.children.append(self)

    def backward(self, grad):
        # self.x.backward(grad @ self.y.data.swapaxes(-1, -2))
        # self.y.backward(self.x.data.swapaxes(-1, 2) @ grad)
        print(grad.shape)
        print(self.x.data.shape)
        print(self.y.data.shape)
        self.x.backward(grad @ self.y.data.T)
        self.y.backward(self.x.data.T @ grad)

class transpose:
    def __init__(self, x, module):
        self.module = module
        self.x = x
        self.operation = self

        self.z = tensor(x.data.T, operation=self)

        x.children.append(self)

    def backward(self, grad):
        self.x.backward(grad.T)

class inverse:
    def __init__(self, x, module):
        self.module = module
        self.x = x
        self.operation = self

        self.z = tensor(self.module.linalg.inv(x.data), operation=self)

        x.children.append(self)

    def backward(self, grad):
        self.x.backward(grad @ self.z.data.T)

class autograd(device):
    def __init__(self, data, parents=None, op=None, device="cpu"):
        super().__init__(device)

        self.data = data
        self.grad = 0
        self.parents = parents or []
        self.op = op
    
    def backward(self, grad=1.0):
        self.grad += grad
        for parent, back_func in self.parents:
            parent.backward(grad * back_func(self))
    
    def __add__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data + other.data,
            parents=[(self, lambda _: 1), (other, lambda _: 1)], op='+', device=self.device)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data * other.data,
            parents=[
                (self, lambda v: other.data),
                (other, lambda v: self.data)],
            op='*', device=self.device)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data ** other.data,
            parents=[
                (self, lambda v: other.data * self.data**(other.data-1)),
                (other, lambda v: self.data**other.data * self.module.log(self.data))],
            op='**', device=self.device)
    
    def __rpow__(self, other):
        return autograd(other) ** self
    
    def __neg__(self):
        return autograd(-self.data, parents=[(self, lambda _: -1)], op='-', device=self.device)
    
    def __sub__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data - other.data,
            parents=[(self, lambda _: 1), (other, lambda _: -1)], op='-', device=self.device)
    
    def __rsub__(self, other):
        return autograd(other) - self
    
    def __truediv__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data / other.data,
            parents=[
                (self, lambda v: 1/other.data),
                (other, lambda v: -self.data/other.data**2)],
            op='/', device=self.device)
    
    def __rtruediv__(self, other):
        return autograd(other) / self

    def __repr__(self):
        return f"autograd(data={self.data}, grad={self.grad})"
    
def ones(shape, device="cpu"):
    return tensor(np.ones(shape), device=device)

def zeros(shape, device="cpu"):
    return tensor(np.zeros(shape), device=device)

def arange(*args):
    if len(args) != 3:
        raise ValueError("arange() takes 3 arguments (start, stop, step)")
    
    list_obj = [i for i in range(args[0], args[1], args[2])]
    return tensor(list_obj)

def parameter(data, device="cpu"):
    return tensor(data, device=device)

def sum(tensor, axis=None):
    return tensor.sum(axis=axis)

def mean(tensor, axis=None):
    return tensor.mean(axis=axis)

def hstack(*tensors):
    return tensor(tensors[0].device.hstack([i.data for i in tensors]))

def vstack(*tensors):
    return tensor(tensors[0].device.vstack([i.data for i in tensors]))

def exp(tensor):
    return tensor.exp()

def log(tensor):
    return tensor.log()

def transpose(tensor):
    return tensor.T()

def inverse(tensor):
    return tensor.inverse()

def matmul(tensor1, tensor2):
    return tensor1 @ tensor2

def pow(tensor1, tensor2):
    return tensor1 ** tensor2

def neg(tensor):
    return tensor.neg()

def div(tensor1, tensor2):
    return tensor1 / tensor2

def sub(tensor1, tensor2):
    return tensor1 - tensor2

def add(tensor1, tensor2):
    return tensor1 + tensor2