import numpy as np
import cv2
import time
import copy    

from visibility_graph import VisibilityGraph
from visibility import Visibility


class OccupancyMap():
    def __init__(self, visualization) -> None:

        # update parameters
        self._alpha = 1.0 # learning rate of new measurements
        self._gamma = 1.0 # discount factor of old measurements

        # map parameters
        self._range_max = 2 # maximum range of the sensor in meters
        self._res = 0.01 # resolution of the map in meters 
        self._threshold = 0.5 # threshold for occupied space
        object_extention = 0.15 # extention of the objects in meters
        self._kernel_size = int(round(2*object_extention / self._res, 0)) # size of the kernel for morphological operations (must be odd)
        self._contour_approx = 0.02 # approximation of the contours
        self._size_x = (0, 5) # map x limits in meters
        self._size_y = (0, 3) # map y limits in meters
        self._boarder_size = int(round(0.1/self._res, 0)) # size of the boarder in pixels
        self._polygon_filter_dist = 0.8/self._res # filter polygons with a distance smaller than this value

        # initialize map and boarder
        self._map = - np.ones((self._pos2idx(self._size_x[1], dim="x"), 
                               self._pos2idx(self._size_y[1], dim="y")))
        self._boarder = [(self._boarder_size,self._boarder_size), 
                         (self._map.shape[0]-self._boarder_size,self._boarder_size),
                         (self._map.shape[0]-self._boarder_size,self._map.shape[1]-self._boarder_size), 
                         (self._boarder_size,self._map.shape[1]-self._boarder_size)]

        # initialize visibility graph
        self.vg = VisibilityGraph(visualization=visualization)  
        self.vis = Visibility()  

        self._polygons = [[]]
        self._path = []
        self._in_polygon_idx = None          

        if visualization:
            import multiprocessing
            from multiprocessing import shared_memory
            from show_map import ShowMap   

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
            x_ticks = np.linspace(0, self._map.shape[0], 11)
            y_ticks = np.linspace(0, self._map.shape[1], 7)
            x_labels = np.linspace(self._size_x[0], self._size_x[1], 11)
            y_labels = np.linspace(self._size_y[0], self._size_y[1], 7)
            labels = (x_ticks, y_ticks, x_labels, y_labels)
            self.process = ShowMap(event=self.event, img_init=img_init, labels=labels)
            self.process.start()





    def __del__(self):
        # free shared memory
        self.shm.close()
        self.shm.unlink()
    
    
    
    def drawMap(self, sensor_data, real_command, desired_command):
        import skimage

        polygons = copy.deepcopy(self._polygons)
        path = copy.copy(self._path)

        # create image and copy map
        map_img = np.ones(self.img.shape, dtype=np.uint8) * 255
        map_array = np.transpose(self._map).copy()

        # # draw map_array in grayscale
        map_array = np.clip(map_array, 0, 1)
        map_img[:, :, 0] = (1-map_array) * 255
        map_img[:, :, 1] = (1-map_array) * 255
        map_img[:, :, 2] = (1-map_array) * 255
        
        # draw polygons in green (or orange if current position is in polygon)
        for idx, poly in enumerate(polygons):
            for i in range(len(poly)):
                if i == len(poly)-1:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[0][0], poly[0][1])        
                else:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
                rr = np.clip(rr, 0, self._map.shape[0]-1)
                cc = np.clip(cc, 0, self._map.shape[1]-1)
                if self._in_polygon_idx == idx: # orange
                    map_img[cc, rr, 0] = 255
                    map_img[cc, rr, 1] = 165
                    map_img[cc, rr, 2] = 0
                else: # green
                    map_img[cc, rr, 0] = 0
                    map_img[cc, rr, 1] = 255
                    map_img[cc, rr, 2] = 0

        # draw polygon center in orange if current position is inside polygon
        if self._in_polygon_idx is not None:  
            x_mean = np.mean([p[0] for p in polygons[self._in_polygon_idx]], dtype=np.uint32)
            y_mean = np.mean([p[1] for p in polygons[self._in_polygon_idx]], dtype=np.uint32)
            rr, cc = skimage.draw.ellipse(x_mean, y_mean, r_radius=4, c_radius=4, shape=self._map.shape)
            map_img[cc, rr, 0] = 255
            map_img[cc, rr, 1] = 165
            map_img[cc, rr, 2] = 0

        # draw path in violett
        for i in range(len(path)-1):
            # print(f"_drawMap: path[i]: {path[i]}, path[i+1]: {path[i+1]}")
            rr, cc = skimage.draw.line(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            rr = np.clip(rr, 0, self._map.shape[0]-1) 
            cc = np.clip(cc, 0, self._map.shape[1]-1)
            map_img[cc, rr, 0] = 238
            map_img[cc, rr, 1] = 130
            map_img[cc, rr, 2] = 238       

        # draw drone position and real velocity in blue
        x0 = self._pos2idx(sensor_data['x_global'], "x")
        y0 = self._pos2idx(sensor_data['y_global'], "y")
        x1 = self._pos2idx(sensor_data['x_global'] + real_command[0], "x")
        y1 = self._pos2idx(sensor_data['y_global'] + real_command[1], "y")
        rr_line, cc_line = skimage.draw.line(x0, y0, x1, y1)
        rr_circle, cc_circle, = skimage.draw.ellipse(x0, y0, r_radius=4, c_radius=4, shape=self._map.shape)
        rr = np.concatenate((rr_line, rr_circle))
        cc = np.concatenate((cc_line, cc_circle))
        map_img[cc, rr, 0] = 0
        map_img[cc, rr, 1] = 0
        map_img[cc, rr, 2] = 255

        # draw desired velocity in red
        x1 = self._pos2idx(sensor_data['x_global'] + desired_command[0], "x")
        y1 = self._pos2idx(sensor_data['y_global'] + desired_command[1], "y")
        rr, cc = skimage.draw.line(x0, y0, x1, y1)
        map_img[cc, rr, 0] = 255
        map_img[cc, rr, 1] = 0
        map_img[cc, rr, 2] = 0

        # save img
        self.img[:] = map_img[:]

        # trigger plotting event
        self.event.set()


    def updateMap(self, sensor_data):
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
            
        # make sure the map is in the correct range
        if np.min(map_array) < -1 or np.max(map_array) > 1:
            raise ValueError('The map is not in the correct range!')
        
        # save the map
        self._map[:] = map_array[:]
        
    def findPath(self, sensor_data, goal):
        # time_s = time.time()

        # convert the start and goal position to indices
        start = (self._pos2idx(sensor_data['x_global'], "x"), self._pos2idx(sensor_data['y_global'], "y"))
        goal = (self._pos2idx(goal[0], "x"), self._pos2idx(goal[1], "y"))

        # copy the map and transpose it (because cv2 takes y-axis in 1. and x-axes in 2. dimension)
        map_array = self._map.copy().astype(np.float32).transpose()

        # Threshold the map
        _, thresholded_map = cv2.threshold(map_array, self._threshold, 1, cv2.THRESH_BINARY)

        # Dilate the obstacles on the map
        dilated_map =   cv2.dilate(
                            src = thresholded_map, 
                            kernel = np.ones((self._kernel_size, self._kernel_size), np.uint8), 
                            iterations = 1
                        )
        
        # check if goal is in obstacle
        goal_in_polygon = False
        if dilated_map[goal[1], goal[0]] == 1:
            goal_in_polygon = True

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

        # print(f"Time to find polygons: {time.time()-time_s:.3f}")
        # time_s = time.time()

        # filter polygons for better performance
        polygons = self._filterPolygons(polygons=polygons, start=start, goal=goal)   

        # add boarder to polygons
        polygons.append(self._boarder)
        
        # check if start is in any polygon (except the boarder)
        if dilated_map[start[1], start[0]] == 1:
            # move out of polygon
            path = self._moveOutPolygon(polygons, start)
        else:
            # Create visibility graph and calculate shortest path
            self.vg.buildGraph(polygons=polygons, start=start, goal=goal)
            path = self.vg.findShortestPath()

            # reset inside polygon index for drawing
            self._in_polygon_idx = None

        # print(f"Time to find path: {time.time()-time_s:.3f}")

        # # convert polygons to pyvisgraph points
        # vg_polygons = []
        # for poly in polygons:
        #     vg_polygons.append([vg.Point(p[0], p[1]) for p in poly])

        # # Get shortest path between two points
        # graph = vg.VisGraph()
        # graph.build(vg_polygons)
        # vg_path = graph.shortest_path(vg.Point(start[0], start[1]), vg.Point(goal[0], goal[1]))
        # path = [(int(round(p.x, 0)), int(round(p.y, 0))) for p in vg_path]

        # calc desired theta (angle of moevement)
        desired_theta = self._calcDesiredTheta(sensor_data=sensor_data, path=path)
        
        # save polygons and path for plotting
        self._polygons = polygons
        self._path = path
        
        return desired_theta, goal_in_polygon
    
    def _moveOutPolygon(self, polygons, start):
        # find polygon that contains start
        idx = self.vis.insidePolygon(polygons=polygons, point=start)     
        
        # calculate mean of closest polygon
        x_mean = np.mean([p[0] for p in polygons[idx]], dtype=np.uint32)
        y_mean = np.mean([p[1] for p in polygons[idx]], dtype=np.uint32)

        # set goal that drone moves away from polygon
        goal = (np.clip(2*start[0]-x_mean, 0, self._map.shape[0]), np.clip(2*start[1]-y_mean, 0, self._map.shape[1]))
        path = [start, goal]

        # set inside polygon index for drawing
        self._in_polygon_idx = idx
        return path
    
    def _filterPolygons(self, polygons, start, goal):
        
        # calculate angle between goal and start to rotate ellipse
        alpha = np.arctan2(goal[1]-start[1], goal[0]-start[0])
        x_ellipse = start[0] + np.cos(alpha)*self._polygon_filter_dist/2
        y_ellipse = start[1] + np.sin(alpha)*self._polygon_filter_dist/2
        
        # filter polygons
        filtered_polygons = []
        for poly in polygons:
            # calculate mean of polygon
            x_poly = np.mean([p[0] for p in poly], dtype=np.uint32)
            y_poly = np.mean([p[1] for p in poly], dtype=np.uint32)

            # # rotate frame
            # dx = x_poly - x_ellipse
            # dy = y_poly - y_ellipse
            # dx_rot = np.cos(alpha)*dx + np.sin(alpha)*dy
            # dy_rot = np.sin(alpha)*dx + np.cos(alpha)*dy

            # # check if polygon is inside rotated ellipse
            # if (dx_rot/150)**2 + (dy_rot/100)**2 <= 1:
            #     filtered_polygons.append(poly)

            # check if polygon is close enough
            if np.sqrt((x_poly-x_ellipse)**2 + (y_poly-y_ellipse)**2) < self._polygon_filter_dist:
                filtered_polygons.append(poly)        
        
        return filtered_polygons
    
    
    def _calcDesiredTheta(self, sensor_data, path):
        # return None if no path was found
        if len(path) <= 1:
            return None
        
        # determine direction of movement
        next_pos = (self._idx2pos(path[1][0], "x"), self._idx2pos(path[1][1], "y"))
        desired_theta = np.arctan2(next_pos[1]-sensor_data["y_global"], next_pos[0]-sensor_data["x_global"])

        return desired_theta
    
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