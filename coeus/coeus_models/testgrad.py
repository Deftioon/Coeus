import numpy as np
import cupy as cp

class autograd:
    def __init__(self, data, parents=None, op=None, device="cpu"):
        if device == "cpu":
            self.device = np
        elif device == "gpu":
            self.device = cp
        else:
            raise ValueError("Unsupported device. Use 'cpu' or 'gpu'.")

        self.data = self.device.array(data)
        self.grad = self.device.zeros(self.data.shape)
        self.parents = parents or []
        self.op = op
    
    def backward(self, grad=1.0):
        self.grad += grad
        for parent, local_grad in self.parents:
            if parent.op == "@":
                parent.backward(grad @ local_grad(self))
            else:
                parent.backward(grad * local_grad(self))
    
    def transpose(self):
        return autograd(
            self.data.transpose(),
            parents=[(self, lambda v: v.transpose())],
            op='transpose'
        )
    
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
    
    def __matmul__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data @ other.data,
            parents=[
                (self, lambda v: other.data.T),
                (other, lambda v: self.data.T)
            ],
            op='@'
        )
    
    def __pow__(self, other):
        other = other if isinstance(other, autograd) else autograd(other)
        return autograd(
            self.data ** other.data,
            parents=[
                (self, lambda v: other.data * self.data**(other.data-1)),
                (other, lambda v: self.data**other.data * np.log(self.data))],
            op='**')
    
    def __neg__(self):
        return autograd(-self.data, parents=[(self, lambda _: -1)], op='-')
    
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

if __name__ == "__main__":
    x = autograd([1,2,3])
    y = autograd([0.5,0.5,0.5])
    z = x ** y
    z.backward()
    print(x.grad)