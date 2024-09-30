import numpy as np
import cupy as cp

class autograd:
    def __init__(self, data, parents=None, op=None, device="cpu"):
        if device == "cpu":
            self.device = "cpu"
            self.module = np
            self.data = np.array(data)
            self.grad = np.zeros(self.data.shape)
        elif device == "gpu":
            self.device = "gpu"
            self.module = cp
            self.data = cp.array(data)
            self.grad = cp.zeros(self.data.shape)
        else:
            raise ValueError("Unsupported device. Use 'cpu' or 'gpu'.")

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
    
    def __matmul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data @ other.data,
            parents=[
                (self, lambda v: other.data),
                (other, lambda v: self.data)
            ],
            op='@', device=self.device
        )
    
    def __rmatmul__(self, other):
        return self.__matmul__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data ** other.data,
            parents=[
                (self, lambda v: other.data * self.data**(other.data-1)),
                (other, lambda v: self.data**other.data * np.log(self.data))],
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