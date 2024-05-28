import matplotlib.pyplot as plt
import numpy as np

# in ms
x = np.linspace(0, 1000, 2600)
frequency = 440
samplingRate = 44100
buffer = (np.sin(2 * np.pi * np.arange(440) * 440 / 44100)).astype(np.float32)

y = np.sin(2*np.pi*44*(x/1000))

plt.plot(x, y)
plt.grid()
plt.show()