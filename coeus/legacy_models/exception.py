class ShapeError(Exception):
    def __init__(self, shape1, shape2):
        self.shape1 = shape1
        self.shape2 = shape2
        self.message = f"Input shape {shape1} does not match model input shape {shape2}"
        super().__init__(self.message)

class ActivationError(Exception):
    def __init__(self, activation, supported_activations):
        self.activation = activation
        self.supported_activations = supported_activations
        self.message = f"Activation function {activation} not supported. Supported functions are {supported_activations}"
        super().__init__(self.message)