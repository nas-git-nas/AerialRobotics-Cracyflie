import numpy as np
import matplotlib.pyplot as plt
from threading import Thread
import time


class ShowMap(Thread):
    def __init__(self, map_init):
        super(ShowMap, self).__init__()
        self.daemon = True
        self.cancelled = False

        self._new_map = False
        self._map = np.copy(map_init)
        
        plt.ion()
        self._fig1, self._ax1 = plt.subplots()

        array = np.transpose(self._map)
        self._grey_img = self._ax1.imshow(array, vmin=0, vmax=1, cmap='gray', origin='lower')
        self._unexplored_img = self._ax1.imshow(array, vmin=0, vmax=1, cmap='Blues', origin='lower', alpha=0.2)

        # plt.xticks(ticks=np.arange(0, self._map.shape[0], 0.5/self._res), 
        #            labels=self._idx2pos(np.arange(0, self._map.shape[0], 0.5/self._res), dim='x'))
        # plt.yticks(ticks=np.arange(0, self._map.shape[1], 0.5/self._res), 
        #            labels=self._idx2pos(np.arange(0, self._map.shape[1], 0.5/self._res), dim='y'))
        # plt.xlabel('x [m]')
        # plt.ylabel('y [m]')

    def setMap(self, map_update):
        self._map = map_update
        self._new_map = True

    def run(self):
        """Overloaded Thread.run, runs the update 
        method once per every 10 milliseconds."""

        while not self.cancelled:
            if self._new_map:
                self.update()
                self._new_map = False
            time.sleep(0.03)

    def cancel(self):
        """End this timer thread"""
        self.cancelled = True

    def update(self, plot_map):
        # transpose the map to match the coordinate system
        plot_map = np.transpose(np.copy(plot_map))

        # plot grey scale map
        grey_map = np.copy(plot_map)
        grey_map[grey_map==-1] = 1
        # plt.imshow(grey_map, vmin=0, vmax=1, cmap='gray', origin='lower')
        self._grey_img.set_data(grey_map)

        # plot unexplored map
        # plt.imshow(-np.clip(map_plot, -1, 0), vmin=0, vmax=1, cmap='Blues', origin='lower', alpha=0.2)
        self._unexplored_img.set_data(-np.clip(plot_map, -1, 0))

        self._fig1.canvas.flush_events()