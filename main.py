import coeus.coeus as coeus

tensor1 = coeus.linalg.tensor([1, 2, 3])
print(tensor1)

tensor2 = coeus.linalg.tensor([4, 5, 6])
print(tensor2)

tensor3 = tensor1 * tensor2
print(tensor3)

tensor3.backward()
print(tensor2.grad)