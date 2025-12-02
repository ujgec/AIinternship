import numpy as np
import matplotlib.pyplot as plt

ones = np.ones((5, 5))

ones[:]=0
ones[:, 2] = ones[:, 2]=1

print(ones)

plt.imshow(ones, cmap='gray_r')

plt.show()
