from .devices import device

class AutogradTensor(device):
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
        return f"AutogradTensor({self.data}, requires_grad={self.requires_grad})"
    
    def backward(self, grad=None, z=None):
        if not self.requires_grad:
            return "False"
        
        if grad is None:
            grad = self.module.ones_like(self.data)
        
        self.grad += grad

        if z is not None:
            self.children.remove(z)
        
        if self.operation:
            if not self.children:
                self.operation.backward(self.grad, self)

    def zero_grad(self):
        self.grad = self.module.zeros_like(self.data)
    
    def zero_grads(self):
        self.grad = self.module.zeros_like(self.data)
        if self.operation:
            for parent in self.operation.parents:
                parent.zero_grads()
    
    def __add__(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return Add().forward(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return Negate().forward(self)
    
    def __sub__(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return Add().forward(self, Negate().forward(other))
    
    def __rsub__(self, other):
        return self.__sub__(other)
    
    def __isub__(self, other):
        return self.__sub__(other)
    
    def __mul__(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return Multiply().forward(self, other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return Divide().forward(self, other)
    
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __itruediv__(self, other):
        return self.__truediv__(other)
    
    def __pow__(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return Power().forward(self, other)
    
    def __rpow__(self, other):
        return other.__pow__(self)
    
    def exp(self):
        return Exp().forward(self)
    
    def log(self):
        return Log().forward(self)
    
    def transpose(self, *dims):
        return Transpose().forward(self, *dims)
    
    def dot(self, other):
        other = other if isinstance(other, AutogradTensor) else AutogradTensor(other)
        return MatMul().forward(self, other)
    
    def __matmul__(self, other):
        return self.dot(other)   

class Parameter(AutogradTensor):
    def __init__(self, data=None, requires_grad=True, operation=None, device="cpu"):
        super().__init__(data, requires_grad, operation, device)

class Add:
    def forward(self, x, y):
        self.x = x
        self.y = y

        requires_grad = x.requires_grad or y.requires_grad

        z = AutogradTensor(x.data + y.data, requires_grad=requires_grad, operation=self)

        self.parents = [x, y]

        z.children.append(x)
        z.children.append(y)
        
        return z
    
    def backward(self, grad, z):
        
        if self.x.requires_grad:
            self.x.backward(grad, z)
        
        if self.y.requires_grad:
            self.y.backward(grad, z)

class Negate:
    def forward(self, x):
        self.x = x

        z = AutogradTensor(-x.data, requires_grad=x.requires_grad, operation=self)

        self.parents = [x]

        z.children.append(x)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(-grad, z)

class Multiply:
    def forward(self, x, y):
        self.x = x
        self.y = y

        requires_grad = x.requires_grad or y.requires_grad

        z = AutogradTensor(x.data * y.data, requires_grad=requires_grad, operation=self)

        self.parents = [x, y]

        z.children.append(x)
        z.children.append(y)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad * self.y.data, z)
        
        if self.y.requires_grad:
            self.y.backward(grad * self.x.data, z)
    
class Divide:
    def forward(self, x, y):
        self.x = x
        self.y = y

        requires_grad = x.requires_grad or y.requires_grad

        z = AutogradTensor(x.data / y.data, requires_grad=requires_grad, operation=self)

        self.parents = [x, y]

        z.children.append(x)
        z.children.append(y)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad / self.y.data, z)
        
        if self.y.requires_grad:
            self.y.backward(-grad * self.x.data / (self.y.data ** 2), z)

class Power:
    def forward(self, x, y):
        self.x = x
        self.y = y

        requires_grad = x.requires_grad or y.requires_grad

        z = AutogradTensor(x.data ** y.data, requires_grad=requires_grad, operation=self)

        self.parents = [x, y]

        z.children.append(x)
        z.children.append(y)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad * self.y.data * (self.x.data ** (self.y.data - 1)), z)
        
        if self.y.requires_grad:
            self.y.backward(grad * (self.x.data ** self.y.data) * self.module.log(self.x.data), z)

class MatMul:
    def forward(self, x, y):
        self.x = x
        self.y = y

        requires_grad = x.requires_grad or y.requires_grad

        z = AutogradTensor(self.module.dot(x.data, y.data), requires_grad=requires_grad, operation=self)

        self.parents = [x, y]

        z.children.append(x)
        z.children.append(y)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(self.module.dot(grad, self.y.data.swapaxes(-1,-2)), z)
        
        if self.y.requires_grad:
            self.y.backward(self.module.dot(self.x.data.swapaxes(-1,2), grad), z)

class Exp:
    def forward(self, x):
        self.x = x

        z = AutogradTensor(self.module.exp(x.data), requires_grad=x.requires_grad, operation=self)

        self.parents = [x]

        z.children.append(x)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad * self.x.data, z)

class Log:
    def forward(self, x):
        self.x = x

        z = AutogradTensor(self.module.log(x.data), requires_grad=x.requires_grad, operation=self)

        self.parents = [x]

        z.children.append(x)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad / self.x.data, z)

class Transpose:
    def forward(self, x, *dims):
        self.x = x
        self.dims = dims

        z = AutogradTensor(self.module.transpose(x.data), requires_grad=x.requires_grad, operation=self)

        self.parents = [x]

        z.children.append(x)

        return z
    
    def backward(self, grad, z):
        if self.x.requires_grad:
            self.x.backward(grad.swapaxes(*self.dims), z)

class autograd(device):
    def __init__(self, data, parents=None, op=None, device="cpu"):
        super().__init__(device)

        self.data = self.module.array(data)
        self.grad = self.module.zeros(self.data.shape)
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