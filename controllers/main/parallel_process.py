import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from multiprocessing import shared_memory


class ParallelProcess(multiprocessing.Process):
    def __init__(self, event, map_init):
        multiprocessing.Process.__init__(self)

        # event is used to trigger the plotter
        self.event = event

        # initial map
        self._map_init = map_init

    def __del__(self): # TODO
        print("destructuring2222222")

    def run(self):
        # get shared memory
        existing_shm = shared_memory.SharedMemory(name='map_current2')
        map_current = np.ndarray(self._map_init.shape, dtype=self._map_init.dtype, buffer=existing_shm.buf)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        img = np.transpose(np.copy(map_current))
        img_explored = ax.imshow(img, vmin=0, vmax=1, cmap='gray', origin='lower')
        img_unexplored = ax.imshow(img, vmin=0, vmax=1, cmap='Blues', origin='lower', alpha=0.2)
        
        while True:
            # Wait for the event to be set
            self.event.wait()

            img = np.transpose(np.copy(map_current))

            # Plot the image using Matplotlib
            img_grey = np.copy(img)
            img_grey[img_grey==-1] = 1
            img_explored.set_data(img_grey)

            img_blue = np.copy(img)
            img_blue = -np.clip(img_blue, -1, 0)
            img_unexplored.set_data(img_blue)

            # display the plot
            plt.pause(0.01)

            # Reset the event
            self.event.clear()

