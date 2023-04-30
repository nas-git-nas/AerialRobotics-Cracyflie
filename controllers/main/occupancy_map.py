import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
import pyvisgraph as vg
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
        self._alpha = 1.0 # learning rate of new measurements
        self._gamma = 1.0 # discount factor of old measurements

        # map processing parameters
        self._threshold = 0.5 # threshold for occupied space
        self._kernel_size = 40 # size of the kernel for morphological operations
        self._contour_approx = 0.01 # approximation of the contours

        # initialize map and boarder
        self._map = - np.ones((self._pos2idx(self._size_x[1]-self._size_x[0], dim="x"), 
                               self._pos2idx(self._size_y[1]-self._size_y[0], dim="y")))
        self._boarder = - np.ones_like(self._map)
        self._boarder[0, :] = 1
        self._boarder[-1, :] = 1
        self._boarder[:, 0] = 1
        self._boarder[:, -1] = 1

        # initialize image for visualization
        img_init = 255 * np.ones((self._map.shape[1], self._map.shape[0], 3), dtype=np.uint8)

        # create shared memory
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=img_init.nbytes, name='map')
            print("Create shared memory")
        except:
            self.shm = shared_memory.SharedMemory(name='map')
            print("Shared memory already exists -> open it")
            # self.shm.close()
            # self.shm.unlink()
            # self.shm = shared_memory.SharedMemory(create=True, size=map_init.nbytes, name='map')

            # if self.shm.size != map_init.nbytes:
            #     print("Error: shared memory has wrong size")
            #     print(f"    shm.size: {self.shm.size}, a.nbytes: {map_init.nbytes}")
            #     exit(1)

        # self.shm = shared_memory.SharedMemory(create=True, size=map_init.nbytes, name='map_current')
        self.img = np.ndarray(img_init.shape, dtype=img_init.dtype, buffer=self.shm.buf)
        self.img[:] = img_init[:]

        # create parallel process and event
        self.event = multiprocessing.Event()
        x_ticks = np.round(np.linspace(0, self._map.shape[0], 12), 1)
        y_ticks = np.round(np.linspace(0, self._map.shape[1], 8), 1)
        x_labels = np.round(np.linspace(self._size_x[0], self._size_x[1], 12), 1)
        y_labels = np.round(np.linspace(self._size_y[0], self._size_y[1], 8), 1)
        labels = (x_ticks, y_ticks, x_labels, y_labels)
        print(labels)
        self.process = ShowMap(event=self.event, img_init=img_init, labels=labels)
        self.process.start()

    def step(self, sensor_data, goal):
        # update the map with the new sensor data
        self._updateMap(sensor_data=sensor_data)

        # find the shortest path from the current position to the goal
        start = (self._pos2idx(sensor_data['x_global'], "x"), self._pos2idx(sensor_data['y_global'], "y"))
        goal = (self._pos2idx(goal[0], "x"), self._pos2idx(goal[1], "y"))
        polygons, path = self._findPath(start=start, goal=goal)

        # draw map to image
        self._drawMap(polygons=polygons, path=path, sensor_data=sensor_data)
        
        # trigger plotting event
        self.event.set()


    def _updateMap(self, sensor_data):
        # copy the map
        map_array = self._map.copy()

        # measurements
        measurements = (sensor_data['range_front'], sensor_data['range_left'], 
                        sensor_data['range_back'], sensor_data['range_right'])
        
        # create measurement map
        meas_map = - np.ones_like(map_array)
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
                meas_map[idx_x[:-1], idx_y[:-1]] = 0
                meas_map[idx_x[-1], idx_y[-1]] = 1
            else:
                meas_map[idx_x, idx_y] = 0

        # update the map with the measurement map
        all_defined = (map_array>=0) & (meas_map>=0) # measurement and map are defined
        map_array[all_defined] = self._alpha*meas_map[all_defined] + (1-self._alpha)*self._map[all_defined]
        meas_defined = (map_array<0) & (meas_map>=0) # only measurement is defined
        map_array[meas_defined] = meas_map[meas_defined]

        # discount the map
        map_array[map_array>=0] = self._gamma*map_array[map_array>=0]

        # TODO: add boarder
        # # add boarder 
        # map_array[self._boarder==1] = 1
            
        # make sure the map is in the correct range
        if np.min(map_array) < -1 or np.max(map_array) > 1:
            raise ValueError('The map is not in the correct range!')
        
        # save the map
        self._map[:] = map_array[:]
        
    def _findPath(self, start, goal):
        # copy the map
        map_array = self._map.copy().astype(np.float32)

        # Threshold the map
        thresholded_map = cv2.threshold(map_array, self._threshold, 1, cv2.THRESH_BINARY)[1]

        # Dilate the obstacles on the map
        dilated_map =   cv2.dilate(
                            src = thresholded_map, 
                            kernel = np.ones((self._kernel_size, self._kernel_size), np.uint8), 
                            iterations = 1
                        )

        # Extract the contours of the obstacles from the map using opencv
        contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the polygons from the contours using opencv
        polygons = []
        for contour in contours:
            # Approximate the contour with a polygon
            polygon = cv2.approxPolyDP(contour, self._contour_approx * cv2.arcLength(contour, True), True)
            # Convert the polygon to a list of points
            polygon_points = [tuple(point[0]) for point in polygon]
            # Add the polygon to the list of polygons
            polygons.append(polygon_points)

        # convert polygons to pyvisgraph points
        vg_polygons = []
        for poly in polygons:
            vg_polygons.append([vg.Point(p[0], p[1]) for p in poly])

        # Create visibility graph
        graph = vg.VisGraph()
        graph.build(vg_polygons)

        # Get shortest path between two points
        vg_path = graph.shortest_path(vg.Point(start[0], start[1]), vg.Point(goal[0], goal[1]))
        path = [(int(round(p.x, 0)), int(round(p.y, 0))) for p in vg_path]

        return polygons, path
    
    def _drawMap(self, polygons, path, sensor_data):
        # create image and copy map
        map_img = np.ones(self.img.shape, dtype=np.uint8) * 255
        map_array = np.transpose(self._map).copy()

        # # draw map_array in grayscale
        map_array = np.clip(map_array, 0, 1)
        map_img[:, :, 0] = (1-map_array) * 255
        map_img[:, :, 1] = (1-map_array) * 255
        map_img[:, :, 2] = (1-map_array) * 255
        
        # draw polygons in green
        for poly in polygons:
            for i in range(len(poly)):
                if i == len(poly)-1:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[0][0], poly[0][1])        
                else:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
                rr = np.clip(rr, 0, map_img.shape[0]-1)
                cc = np.clip(cc, 0, map_img.shape[1]-1)
                map_img[rr, cc, 0] = 0
                map_img[rr, cc, 1] = 255
                map_img[rr, cc, 2] = 0

        # draw path in red
        for i in range(len(path)-1):
            # print(f"_drawMap: path[i]: {path[i]}, path[i+1]: {path[i+1]}")
            rr, cc = skimage.draw.line(path[i][1], path[i][0], path[i+1][1], path[i+1][0])
            rr = np.clip(rr, 0, map_img.shape[0]-1)
            cc = np.clip(cc, 0, map_img.shape[1]-1)
            map_img[rr, cc, 0] = 255
            map_img[rr, cc, 1] = 0
            map_img[rr, cc, 2] = 0

        # draw heading in blue
        x0 = self._pos2idx(sensor_data['x_global'], "x")
        y0 = self._pos2idx(sensor_data['y_global'], "y")
        x1 = self._pos2idx(sensor_data['x_global'] + 0.1 * np.cos(sensor_data['yaw']), "x")
        y1 = self._pos2idx(sensor_data['y_global'] + 0.1 * np.sin(sensor_data['yaw']), "y")
        rr_line, cc_line = skimage.draw.line(y0, x0, y1, x1)
        rr_circle, cc_circle, = skimage.draw.ellipse(y0, x0, r_radius=4, 
                                                     c_radius=4, shape=map_array.shape)
        rr = np.concatenate((rr_line, rr_circle))
        cc = np.concatenate((cc_line, cc_circle))
        map_img[rr, cc, 0] = 0
        map_img[rr, cc, 1] = 0
        map_img[rr, cc, 2] = 255

        # save img
        self.img[:] = map_img[:]

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