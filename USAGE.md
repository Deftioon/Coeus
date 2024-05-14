# Coeus Usage Docs

## Models

### Univariate Regression

Coeus supports an intuitive platform to perform Univariate Linear Regression, fitting data to functions in the form of $w_1f_1(x) + w_2f_2(x) + ... +w_nf_n(x)+b$. 

**Creating a Model**

The syntax for initializing a model is as follows:

```py
Linear(funcs: list, error: str)
"""
Arguments:
funcs: A list of functions that are to be fit linearly onto a set of data.
error: A string containing the Loss Function the user wishes to use.
"""
```

Where `funcs` is a list of functions predefined to the user to be fit linearly onto a set of data. The `error` is a string containing the Loss Function the user wishes to use. At the moment, only MSE (`"MSE"`) is supported.

Example:

```py
import numpy as np
import coeus.models.regression as regression

model = regression.Linear([lambda x: x, lambda x: x**2], "MSE")
```

**Fitting Data**

To fit the model to some data, use the `fit()` function. The syntax is as follows:

```py
fit(x: np.ndarray, y: np.ndarray)
"""
Arguments:
x: Set of training data
y: Set of target data
"""
```

`x` is the set of training data, and `y` is the set of target data.

Example:

```py
import numpy as np
import coeus.models.regression as regression

model = regression.Linear([lambda x: x, lambda x: x**2], "MSE")

x = np.sort(np.random.rand(100))
y = 0.5 * x ** 2 + 0.1 * x + np.random.rand(100)

model.fit(x, y)
```



**Showing the Model**

By using `print(model)`, a quick summary of the model is showed.

```py
> print(model)
=============MODEL SUMMARY==============
Linear Regression Model:
Size: 1
Error: MSE
===============DETAILS==================
Average Error: 0.0853981092154973
=======================================
```



**Performing Predictions**

After fitting the model with `fit()`, the user can then predict a piece of data with the fitted model.

```py
predict(X: np.ndarray, bayes = False) -> np.ndarray
"""
Arguments:
X: Piece, or array of data
bayes: Toggle to show predictive density or not
"""
```

`X` is a piece of data, or an array of data, to perform predictions on.

The `bayes` argument specifies whether or not to calculate and display the Predictive Density on a piece of data. For this to show, the data must be of shape `(1, 1)`

Example:

```py
import numpy as np
import coeus.models.regression as regression

model = regression.Linear([lambda x: x, lambda x: x**2], "MSE")

x = np.sort(np.random.rand(100))
y = 0.5 * x ** 2 + 0.1 * x + np.random.rand(100)

model.fit(x, y)

test = np.array([10,11,12])
print(model.predict(test))
> [70.33082061  85.19141525 101.48508885]
```



**Performing Forecasts**

Coeus can allow models to perform forecasts on the data, at a set forecast length and forecast steps. The `forecast`syntax is as follows:

```py
forecast(steps: int, increment = 1) -> np.ndarray
"""
Arguments:
steps: Amount of steps in the future to forecast
increment: Interval between steps
"""
```

`forecast` performs forecasting for `steps` steps, each step separated by `increment` intervals.

Example:

```py
import numpy as np
import coeus.models.regression as regression

model = regression.Linear([lambda x: x, lambda x: x**2], "MSE")

x = np.sort(np.random.rand(100))
y = 0.5 * x ** 2 + 0.1 * x + np.random.rand(100)

model.fit(x, y)

print(model.forecast(5, 0.1))
> [1.21347336 1.40943853 1.6282886  1.87002357 2.13464342]
```



**Visualising the Model**

To show the model, use the `show()` function. The syntax is as follows:

```py
show(forecast_steps = 0, forecast_increment = 1, custom_title = "X vs Y", custom_xlabel = "X", custom_ylabel = "Y", save = False)
"""
Arguments:
forecast_steps: Steps to forecast, if needed.
forecast_increment: Increments to forecast, if needed
custom_title: Title for graph
custom_xlabel: x-axis label for graph
custom_ylabel: y-axis label for graph
save: Toggle to save to exports/ or not
"""
```

`show` displays a visualisation for the univariate linear regression, and can save the plot to the `exports` directory.



**Retrieving the Model**

Sample of Univariate Linear Regression in Coeus:

```py
import numpy as np
import coeus.models.regression as regression

# Generate Data
x = np.random.rand(100)
y = x**2 + np.random.rand(100)

# Create Model
model = regression.Linear()
```

