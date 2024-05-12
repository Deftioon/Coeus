import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, funcs: list, error: str):
        supported_funcs = ["MSE"]
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
Regression Model:
Size: {self.paramNums}
Error: {self.error}
===============DETAILS==================
Average Error: {self.average_error}
Parameters: {self.params}
=======================================
            """
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.fit_train = X
        self.fit_target = y
        
        if self.error == "MSE":
            x_train = np.asarray([i(X) for i in self.funcs]).T
            x_train = np.hstack((np.ones([X.shape[0],1], X.dtype), x_train))

            self.params = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y

            self.average_error = np.mean((y - x_train @ self.params) ** 2)
    
    def predict(self, X: np.ndarray):
        x_test = np.asarray([i(X) for i in self.funcs]).T
        x_test = np.hstack((np.ones([X.shape[0],1], X.dtype), x_test))

        return x_test @ self.params

    def forecast(self, steps: int, increment = 1):
        self.forecast_train = np.arange(self.fit_train[-1], self.fit_train[-1] + steps * increment)
        self.forecast_out = self.predict(self.forecast_train)
        return self.forecast_out

    def get_last_forecast(self):
        return self.forecast_train, self.forecast_out
    
    def get_params(self):
        return self.params

    def show(self, forecast_steps = 0, forecast_increment = 1, custom_title = "X vs Y", custom_xlabel = "X", custom_ylabel = "Y", save = False):
        plt.title(custom_title)
        plt.xlabel(custom_xlabel)
        plt.ylabel(custom_ylabel)

        if forecast_steps == 0:
            plt.scatter(self.fit_train, self.fit_target)
            plt.plot(self.fit_train, self.predict(self.fit_train), label = "Fit")

            if save:
                plt.savefig(f"exports/{custom_title}.png")

            plt.legend()

            plt.show()
            return

        else:
            plt.scatter(self.fit_train, self.fit_target)
            plt.plot(self.fit_train, self.predict(self.fit_train), label = "Fit")
            
            forecast = self.forecast(forecast_steps, forecast_increment)
            plt.plot(self.forecast_train, forecast, "--", label = "Forecasted")
            
            plt.legend()

            if save:
                plt.savefig(f"exports/{custom_title}.png")

            plt.show()
            return