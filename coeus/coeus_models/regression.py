from coeus.utils import linalg
from coeus.utils.autograd import tensor

class Linear:
    def __init__(self, funcs: list, error: str):
        supported_funcs = ["MSE", "MAE"]
        if error not in supported_funcs:
            raise NotImplementedError(f"Error function {error} not supported. Supported functions are {supported_funcs}")
        
        self.paramNums = len(funcs)
        self.funcs = funcs
        self.error = error
        self.average_error = "Not Calculated"
        self.params = "Not Calculated"

    def __str__(self):
        return f"""
=============MODEL SUMMARY==============
Linear Regression Model:
Size: {self.paramNums}
Error: {self.error}
===============DETAILS==================
Average Error: {self.average_error}
=======================================
            """
    
    def fit(self, X: tensor, y: tensor):
        self.fit_train = X
        self.fit_target = y
        
        if self.error == "MSE":
            x_train = tensor([i(X) for i in self.funcs]).T
            x_train = linalg.hstack(linalg.ones([X.shape[0], 1]), x_train)

            self.params = linalg.inverse(x_train.T @ x_train) @ x_train.T @ y

            self.average_error = linalg.mean((y - x_train @ self.params) ** 2)
    
    def predict(self, X: tensor, bayes=False):
        if not bayes:
            x_test = tensor([i(X) for i in self.funcs]).T
            x_test = linalg.hstack(linalg.ones([X.shape[0], 1]), x_test)

            return x_test @ self.params
        
        elif bayes:
            pass
    
    def forecast(self, steps: int, increment=1):
        self.forecast_train = linalg.arange(self.fit_train[-1], self.fit_train[-1] + steps * increment, increment)
        self.forecast_out = self.predict(self.forecast_train)
        return self.forecast_out
    
    def get_last_forecast(self):
        return self.forecast_train, self.forecast_out
    
    def get_params(self):
        return self.params