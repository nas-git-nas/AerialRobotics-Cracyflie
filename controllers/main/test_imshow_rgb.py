import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# blue
array = np.concatenate((np.zeros((5,5,1)), np.zeros((5,5,1)), np.ones((5,5,1))), axis=2) * 255
img = ax.imshow(array, vmin=0, vmax=255, origin='lower')
plt.pause(1)

# green
array = np.concatenate((np.zeros((5,5,1)), np.ones((5,5,1)), np.zeros((5,5,1))), axis=2) * 255
img.set_data(array)
plt.pause(1)

# red
array = np.concatenate((np.ones((5,5,1)), np.zeros((5,5,1)), np.zeros((5,5,1))), axis=2) * 255
img.set_data(array)
plt.pause(1)

# black
array = np.concatenate((np.zeros((5,5,1)), np.zeros((5,5,1)), np.zeros((5,5,1))), axis=2) * 255
img.set_data(array)
plt.pause(1)

# white
array = np.concatenate((np.ones((5,5,1)), np.ones((5,5,1)), np.ones((5,5,1))), axis=2) * 255
img.set_data(array)
plt.pause(1)