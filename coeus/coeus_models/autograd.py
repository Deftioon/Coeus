import numpy as np
import cupy as cp

# class autograd:
#     def __init__(self, data, parents=None, op=None):
#         self.data = data
#         self.grad = 0.0
#         self.parents = parents or []
#         self.op = op
    
#     def backward(self, grad=1.0):
#         self.grad += grad
#         for parent, local_grad in self.parents:
#             parent.backward(grad * local_grad(self))

#     def __add__(self, other):
#         other = other if isinstance(other, autograd) else autograd(other)
#         return autograd(
#             self.data + other.data,
#             parents=[(self, lambda _: 1), (other, lambda _: 1)], op='+')

#     def __mul__(self, other):
#         other = other if isinstance(other, autograd) else autograd(other)
#         return autograd(
#             self.data * other.data,
#             parents=[
#                 (self, lambda v: other.data),
#                 (other, lambda v: self.data)],
#             op='*')

#     def __sub__(self, other):
#         other = other if isinstance(other, autograd) else autograd(other)
#         return autograd(
#             self.data - other.data,
#             parents=[(self, lambda _: 1), (other, lambda _: -1)], op='-')

#     def __truediv__(self, other):
#         other = other if isinstance(other, autograd) else autograd(other)
#         return autograd(
#             self.data / other.data,
#             parents=[
#                 (self, lambda v: 1/other.data),
#                 (other, lambda v: -self.data/other.data**2)],
#             op='/')

#     def __repr__(self):
#         return f"autograd(data={self.data}, grad={self.grad})"

class autograd:
    def __init__(self, data, parents=None, op=None, device = "cpu"):
        if device == "cpu":
            self.device = np
        
        if device == "gpu":
            self.device = cp

        self.data = self.device.array(data)
        self.grad = self.device.zeros(self.data.shape)
        self.parents = parents or []
        self.op = op
    
    def backward(self, grad=1.0):
        self.grad += grad
        for parent, local_grad in self.parents:
            parent.backward(grad * local_grad(self))
    
    def transpose(self):
        return autograd(self.data.transpose())
    
    def __add__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data + other.data,
            parents=[(self, lambda _: 1), (other, lambda _: 1)], op='+')
    
    def __mul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data * other.data,
            parents=[
                (self, lambda v: other.data),
                (other, lambda v: self.data)],
            op='*')
    
    def __sub__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data - other.data,
            parents=[(self, lambda _: 1), (other, lambda _: -1)], op='-')
    
    def __truediv__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data / other.data,
            parents=[
                (self, lambda v: 1/other.data),
                (other, lambda v: -self.data/other.data**2)],
            op='/')

    def __repr__(self):
        return f"autograd(data={self.data}, grad={self.grad})"
    