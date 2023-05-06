import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory


class VisualizationProcess(multiprocessing.Process):
    def __init__(self, event, img_init, labels):
        multiprocessing.Process.__init__(self)

        # event is used to trigger the plotter
        self.event = event

        # initial map
        self._img_init = img_init
        self._labels = labels

    def run(self):
        # get shared memory
        existing_shm = shared_memory.SharedMemory(name='map')
        map_current = np.ndarray(self._img_init.shape, dtype=self._img_init.dtype, buffer=existing_shm.buf)

        # create plot and draw map
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        img = ax.imshow(map_current, vmin=0, vmax=255, origin='lower')
        x_ticks, y_ticks, x_labels, y_labels = self._labels
        ax.set_xticks(x_ticks, x_labels)
        ax.set_yticks(y_ticks, y_labels)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        
        while True:
            # Wait for the event to be set
            if not self.event.wait(timeout=1):
                # if waited too long -> free shared memory and exit process
                existing_shm.close()
                break

            # update map and display plot
            img.set_data(map_current)
            plt.pause(0.01)

            # Reset the event
            self.event.clear()

        