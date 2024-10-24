import numpy as np
import matplotlib.pyplot as plt

def Vc(voltage, time, capacitance, resistance):
    return voltage * (1 - np.exp(-time / (resistance * capacitance)))

resistance = 1.25
capacitance = 100 * 10**(-6)
voltage = 15
times = np.arange(0, 625 * 10**(-6), 10**(-7))

data = Vc(voltage, times, capacitance, resistance)
plt.plot(times, data)
plt.show()


