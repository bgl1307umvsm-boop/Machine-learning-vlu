import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-8, 8, 200)

sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid, label="Sigmoid")

plt.axhline(y=0.5, linestyle="--", label="y = 0.5")

plt.axvline(x=0, linestyle="--", label="z = 0")

plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Hàm Sigmoid")
plt.grid(True)
plt.legend()

plt.savefig("baitap02/sigmoid.png")

plt.show()