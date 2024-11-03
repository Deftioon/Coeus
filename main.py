import coeus.coeus as coeus

tensor1 = coeus.autograd.tensor([1, 2, 3, 4, 5], requires_grad=True)
tensor2 = coeus.autograd.tensor([1, 2, 3, 4, 5], requires_grad=True)

result = tensor1 * tensor2

result.backward()
print(tensor1.grad)
print(tensor2.grad)

result = tensor1 + tensor2
result.backward()
print(tensor1.grad)
print(tensor2.grad)