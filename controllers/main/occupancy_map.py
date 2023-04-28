import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory
import time
import atexit

# from parallel_process import ParallelProcess
from show_map import ShowMap

# def exit_handler():
#     print('My application is ending!')

# atexit.register(exit_handler)

class OccupancyMap():
    def __init__(self) -> None:
        
        # map parameters
        self._range_max = 2 # maximum range of the sensor in meters
        self._res = 0.01 # resolution of the map in meters      
        self._size_x = (-0.5, 5.5) # map x limits in meters
        self._size_y = (-0.5, 3.5) # map y limits in meters

        # update parameters
        self._alpha = 0.5 # learning rate of new measurements
        self._gamma = 1.0 # discount factor of old measurements

        # initialize map and boarder
        map_init = - np.ones((self._pos2idx(self._size_x[1]-self._size_x[0], dim="x"), 
                               self._pos2idx(self._size_y[1]-self._size_y[0], dim="y")))
        self._boarder = np.ones_like(map_init)
        self._boarder[0, :] = 0
        self._boarder[-1, :] = 0
        self._boarder[:, 0] = 0
        self._boarder[:, -1] = 0

        # create shared memory
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=map_init.nbytes, name='map')
            print("Create shared memory")
        except:
            self.shm = multiprocessing.shared_memory.SharedMemory(name='map')
            print("Shared memory already exists -> open it")
            # self.shm.close()
            # self.shm.unlink()
            # self.shm = shared_memory.SharedMemory(create=True, size=map_init.nbytes, name='map')

            # if self.shm.size != map_init.nbytes:
            #     print("Error: shared memory has wrong size")
            #     print(f"    shm.size: {self.shm.size}, a.nbytes: {map_init.nbytes}")
            #     exit(1)

        # self.shm = shared_memory.SharedMemory(create=True, size=map_init.nbytes, name='map_current')
        self._map = np.ndarray(map_init.shape, dtype=map_init.dtype, buffer=self.shm.buf)
        self._map[:] = map_init[:]

        # create parallel process and event
        self.event = multiprocessing.Event()
        self.process = ShowMap(event=self.event, map_init=map_init)
        self.process.start()


    def update(self, sensor_data):
        # measurements
        measurements = (sensor_data['range_front'], sensor_data['range_left'], 
                        sensor_data['range_back'], sensor_data['range_right'])
        
        # create measurement map
        meas_map = - np.ones_like(self._map)
        for j, meas in enumerate(measurements):
            # calculate yaw: positive yaw is counter clockwise
            yaw_sensor = sensor_data['yaw'] + j*np.pi/2

            # calculate the distance covered by the sensor
            dists_len = np.arange(0, np.minimum(meas, self._range_max), self._res)
            dists_x = sensor_data['x_global'] + dists_len*np.cos(yaw_sensor)
            dists_y = sensor_data['y_global'] + dists_len*np.sin(yaw_sensor)

            # calculate the indices of the map that are covered by the sensor
            idx_x = self._pos2idx(dists_x, dim='x')
            idx_y = self._pos2idx(dists_y, dim='y')

            # add measurement to the measurement map
            if meas < self._range_max:
                meas_map[idx_x[:-1], idx_y[:-1]] = 1
                meas_map[idx_x[-1], idx_y[-1]] = 0
            else:
                meas_map[idx_x, idx_y] = 1

        # update the map with the measurement map
        all_defined = (self._map>=0) & (meas_map>=0) # measurement and map are defined
        self._map[all_defined] = self._alpha*meas_map[all_defined] + (1-self._alpha)*self._map[all_defined]
        meas_defined = (self._map<0) & (meas_map>=0) # only measurement is defined
        self._map[meas_defined] = meas_map[meas_defined]

        # discount the map
        self._map[self._map>=0] = self._gamma*self._map[self._map>=0]

        # add boarder
        self._map[self._boarder==0] = 0
            
        # make sure the map is in the correct range
        if np.min(self._map) < -1 or np.max(self._map) > 1:
            raise ValueError('The map is not in the correct range!')
        
        self.event.set()

    def freeSharedMemory(self): # TODO
        self.shm.close()
        self.shm.unlink()
    
    def _pos2idx(self, pos, dim):
        if dim == "x":
            idx = np.round((pos - self._size_x[0])/self._res, 0).astype(int)
        elif dim == "y":
            idx = np.round((pos - self._size_y[0])/self._res, 0).astype(int)
        else:
            raise ValueError('Dimension not defined!')
        
        if hasattr(self, '_map'):
            if dim == "x":
                idx = np.clip(idx, 0, self._map.shape[0]-1)
            elif dim == "y":
                idx = np.clip(idx, 0, self._map.shape[1]-1)
            else:
                raise ValueError('Dimension not defined!')

        return idx
    
    def _idx2pos(self, idx, dim):
        if dim == "x":
            pos = idx*self._res + self._size_x[0]
        elif dim == "y":
            pos = idx*self._res + self._size_y[0]
        else:
            raise ValueError('Dimension not defined!')
        
        return pos


def test_map():

    occ_map = OccupancyMap()

    for x in np.linspace(0, 5, 100):
        sensor_data = {'range_front': 1.9, 'range_left': 1.5, 'range_back': 2.0, 'range_right': 0.5, 'yaw': 0.0, 'x_global': x, 'y_global': 2}
        occ_map.update(sensor_data)

        time.sleep(0.5)

    occ_map.freeSharedMemory()


if __name__ == "__main__":
    test_map()