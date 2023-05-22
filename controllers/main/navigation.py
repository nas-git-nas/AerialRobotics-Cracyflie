import numpy as np

import time
import copy    
import sys, os
sys.path.append(os.path.abspath("C:/Users/hendr/anaconda3/envs/aerial2/Lib/site-packages"))
import cv2
from scipy.ndimage import binary_dilation, label

try:
    from visibility_graph import VisibilityGraph
    from parameters import Parameters
except:
    from .visibility_graph import VisibilityGraph
    from .parameters import Parameters


class Navigation():
    def __init__(self, params: Parameters) -> None:
        # initialize visibility graph
        self.params = params
        self.vg = VisibilityGraph()  
       
        # initialize map and boarder
        # self._map = - np.ones((self._pos2idx(self.params.map_size_x[1], dim="x"), 
        #                        self._pos2idx(self.params.map_size_y[1], dim="y")))
        self._map = np.zeros((self._pos2idx(self.params.map_size_x[1], dim="x"), 
                               self._pos2idx(self.params.map_size_y[1], dim="y")))
        self._boarder = [(self.params.map_boarder_size,self.params.map_boarder_size), 
                         (self._map.shape[0]-self.params.map_boarder_size,self.params.map_boarder_size),
                         (self._map.shape[0]-self.params.map_boarder_size,self._map.shape[1]-self.params.map_boarder_size), 
                         (self.params.map_boarder_size,self._map.shape[1]-self.params.map_boarder_size)]

        # initialize polygons and path
        self._polygons = [[]]
        self._unfiltered_polygons = [[]]
        self._path = []
        self._start_in_polygon = None          

    def get(self, attr):
        if attr == "map":
            return np.copy(self._map)
        elif attr == "polygons":
            return copy.deepcopy(self._polygons)
        elif attr == "unfiltered_polygons":
            return copy.deepcopy(self._unfiltered_polygons)
        elif attr == "path":
            return copy.copy(self._path)
        elif attr == "start_in_polygon":
            return copy.copy(self._start_in_polygon)
        else:
            raise ValueError("Attribute not found")

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
            dists_len = np.arange(0, np.minimum(meas, self.params.nav_range_max), self.params.map_res)
            dists_x = sensor_data['x_global'] + dists_len*np.cos(yaw_sensor)
            dists_y = sensor_data['y_global'] + dists_len*np.sin(yaw_sensor)

            # calculate the indices of the map that are covered by the sensor
            idx_x = self._pos2idx(dists_x, dim='x')
            idx_y = self._pos2idx(dists_y, dim='y')

            # add measurement to the measurement map
            if meas < self.params.nav_range_max \
                and (len(dists_x) > 0 and self.params.map_size_x[0] < dists_x[-1]) \
                and (len(dists_x) > 0 and dists_x[-1] < self.params.map_size_x[1]) \
                and (len(dists_y) > 0 and self.params.map_size_y[0] < dists_y[-1]) \
                and (len(dists_y) > 0 and dists_y[-1] < self.params.map_size_y[1]):
                meas_map[idx_x[:-3], idx_y[:-3]] = 0 # do not update the last two cells before the obstacle because of measurement noise
                meas_map[idx_x[-1], idx_y[-1]] = 1
            else:
                meas_map[idx_x, idx_y] = 0

        # update the map with the measurement map
        all_defined = (map_array>=0) & (meas_map>=0) # measurement and map are defined
        map_array[all_defined] = self.params.nav_alpha*meas_map[all_defined] + (1-self.params.nav_alpha)*self._map[all_defined]
        meas_defined = (map_array<0) & (meas_map>=0) # only measurement is defined
        map_array[meas_defined] = meas_map[meas_defined]

        # discount the map
        map_array[map_array>=0] = self.params.nav_gamma*map_array[map_array>=0]
            
        # make sure the map is in the correct range
        if np.min(map_array) < -1 or np.max(map_array) > 1:
            raise ValueError('The map is not in the correct range!')
        
        # save the map
        self._map[:] = map_array[:]  
        
    def findPath(self, sensor_data, goal):
        # convert the start and goal position to indices
        start = (self._pos2idx(sensor_data['x_global'], "x"), self._pos2idx(sensor_data['y_global'], "y"))
        goal = (self._pos2idx(goal[0], "x"), self._pos2idx(goal[1], "y"))

        polygons = self._findPolygons() 
        self._unfiltered_polygons = copy.deepcopy(polygons)     

        # add boarder and filter polygons for better performance
        polygons.append(self._boarder)
        polygons, boarder_added = self._filterPolygons(polygons=polygons, start=start, goal=goal)

        # add start to the beginning and goal to the end of the polygons
        polygons = [[start]] + polygons + [[goal]]

        # Create visibility graph and calculate shortest path
        start_in_polygon, goal_in_polygon = self.vg.buildGraph(polygons=polygons, start=start, goal=goal, boarder_added=boarder_added)
        path = self.vg.findShortestPath()

        # calc desired theta (angle of moevement)
        desired_yaw = self._calcDesiredYaw(sensor_data=sensor_data, path=path, polygons=polygons, start_in_poly=start_in_polygon)
        
        # save polygons and path for plotting
        self._polygons = polygons
        self._path = path
        self._start_in_polygon = start_in_polygon
        
        return desired_yaw, goal_in_polygon, start_in_polygon
    
    def _findPolygons(self):
        # copy the map and transpose it (because cv2 takes y-axis in 1. and x-axes in 2. dimension)
        map_array = self._map.copy().astype(np.float32).transpose()

        # Threshold the map
        #thresholded_map = (map_array > self.params.nav_threshold).astype(np.uint8)

        # Threshold the map
        _, thresholded_map = cv2.threshold(map_array, self.params.nav_threshold, 1, cv2.THRESH_BINARY)

        # Dilate the obstacles on the map
        dilated_map =   cv2.dilate(
                           src = thresholded_map, 
                           kernel = np.ones((self.params.nav_kernel_size,self.params.nav_kernel_size), np.uint8), 
                           iterations = 1
                       )
        
        # Extract the contours of the obstacles from the map using opencv
        contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the polygons from the contours using opencv
        polygons = []
        for c in contours:
            c_max = np.max(c, axis=(0,1)) + self.params.nav_object_extention
            c_min = np.min(c, axis=(0,1)) - self.params.nav_object_extention
            polygons.append([(c_min[0], c_min[1]), (c_max[0], c_min[1]), 
                             (c_max[0], c_max[1]), (c_min[0], c_max[1])])

        return polygons
    
    def _filterPolygons(self, polygons, start, goal):   
        # displace center from circle of view into direction of movement
        alpha = np.arctan2(goal[1]-start[1], goal[0]-start[0])
        x_circle = start[0] + np.cos(alpha)*self.params.nav_polygon_filter_dist/2
        y_circle = start[1] + np.sin(alpha)*self.params.nav_polygon_filter_dist/2
        
        # filter all polygons except of the boarder (last element)
        filtered_polygons = []
        for poly in polygons[:-1]:
            # calculate mean of polygon
            poly_mean = np.mean(poly, axis=0, dtype=np.uint32)

            # check if polygon is close enough
            if np.sqrt((poly_mean[0]-x_circle)**2 + (poly_mean[1]-y_circle)**2) < self.params.nav_polygon_filter_dist:
                filtered_polygons.append(poly)

        # filter boarder if it is too close to the start
        boarder_added = False
        if start[0] < self.params.nav_polygon_filter_dist or abs(self._map.shape[0]-start[0]) < self.params.nav_polygon_filter_dist \
            or start[1] < self.params.nav_polygon_filter_dist or abs(self._map.shape[1]-start[1]) < self.params.nav_polygon_filter_dist:
            filtered_polygons.append(polygons[-1])
            boarder_added = True
        
        return filtered_polygons, boarder_added
    
    
    def _calcDesiredYaw(self, sensor_data, path, polygons, start_in_poly):
        # return None if no path was found
        if len(path) <= 1:
            return None
        
        # determine next point
        for p in path:
            next_point = (self._idx2pos(p[0], "x"), self._idx2pos(p[1], "y"))
            if self._dist2point(sensor_data, next_point) > self.params.nav_point_reached_dist:
                break
        
        # calculate desired yaw: desired direction of movement
        desired_yaw = np.arctan2(next_point[1]-sensor_data["y_global"], next_point[0]-sensor_data["x_global"])

        # correct desired yaw if drone is inside a polygon: move outwards/away from obstacle
        if start_in_poly:
            # calculate yaw with respect to polygon
            poly_mean_xy = np.mean(polygons[start_in_poly], axis=0) # in pixels
            mean_x = self._idx2pos(idx=poly_mean_xy[0], dim="x") # in meters
            mean_y = self._idx2pos(idx=poly_mean_xy[1], dim="y") # in meters
            poly_yaw = np.arctan2(mean_y-sensor_data["y_global"], mean_x-sensor_data["x_global"])

            # normalize polygon yaw with respect to desired yaw
            if poly_yaw - desired_yaw > np.pi:
                poly_yaw -= 2*np.pi
            elif poly_yaw - desired_yaw <= -np.pi:
                poly_yaw += 2*np.pi

            # correct yaw to move outwards of polygon
            if desired_yaw > poly_yaw:
                desired_yaw += self.params.nav_yaw_correction
            else:
                desired_yaw -= self.params.nav_yaw_correction

        return desired_yaw
    
    def _pos2idx(self, pos, dim):
        if dim == "x":
            idx = np.round((pos - self.params.map_size_x[0])/self.params.map_res, 0).astype(int)
        elif dim == "y":
            idx = np.round((pos - self.params.map_size_y[0])/self.params.map_res, 0).astype(int)
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
            pos = idx*self.params.map_res + self.params.map_size_x[0]
        elif dim == "y":
            pos = idx*self.params.map_res + self.params.map_size_y[0]
        else:
            raise ValueError('Dimension not defined!')
        
        return pos
    
    def _dist2point(self, sensor_data, point):
        v_x = point[0] - sensor_data['x_global']
        v_y = point[1] - sensor_data['y_global']
        return np.sqrt(v_x**2 + v_y**2)


# def test_map():

    # start = time.time()
    # for _ in range(100):
    #     map_array = np.zeros((500, 300)).astype(np.uint8)
    #     map_array[20:40, 30:60] = 1
    #     map_array[100:200, 100:200] = 1
    #     map_array[300:400, 100:200] = 1

    #     # Threshold the map
    #     thresholded_map = (map_array > 0.9).astype(np.uint8)

    #     # Dilate the obstacles on the map
    #     structure = np.ones((31,31))
    #     dilated_map = binary_dilation(thresholded_map, structure=structure).astype(np.uint8)

    #     # Label the objects in the map
    #     labeled_map, num_objects = label(dilated_map)

    #     # Extract the polygons from the labeled objects
    #     polygons = []
    #     for i in range(1, num_objects+1):
    #         # Find the indices of the object in the map
    #         x_idx = np.where(np.any(labeled_map==i, axis=0))
    #         y_idx = np.where(np.any(labeled_map==i, axis=1))

    #         # define polygon as rectangle around object
    #         polygons.append([(np.min(x_idx), np.min(y_idx)), 
    #                          (np.max(x_idx), np.min(y_idx)), 
    #                          (np.max(x_idx), np.max(y_idx)), 
    #                          (np.min(x_idx), np.max(y_idx))])
    # print(f"Scipy time: {round(time.time()-start,4)}s")

#     start = time.time()
#     for _ in range(1000):
#         map_array = np.zeros((500, 300)).astype(np.uint8)
#         map_array[20:40, 30:60] = 1
#         map_array[100:200, 100:200] = 1
#         map_array[300:400, 100:200] = 1

#         # Threshold the map
#         _, thresholded_map = cv2.threshold(map_array, 0.9, 1, cv2.THRESH_BINARY)

#         # Dilate the obstacles on the map
#         dilated_map =   cv2.dilate(
#                             src = thresholded_map, 
#                             kernel = np.ones((31,31), np.uint8), 
#                             iterations = 1
#                         )
        
#         # Extract the contours of the obstacles from the map using opencv
#         contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Extract the polygons from the contours using opencv
#         polygons = []
#         for contour in contours:
#             # Approximate the contour with a polygon
#             polygon = cv2.approxPolyDP(contour, 0.2 * cv2.arcLength(contour, True), True)
#             # Convert the polygon to a list of points
#             polygon_points = [tuple(point[0]) for point in polygon]
#             # Add the polygon to the list of polygons
#             polygons.append(polygon_points)
#     print(f"Opencv time: {round(time.time()-start,4)}s")

#     start = time.time()
#     for _ in range(1000):
#         map_array = np.zeros((500, 300)).astype(np.uint8)
#         map_array[20:40, 30:60] = 1
#         map_array[100:200, 100:200] = 1
#         map_array[300:400, 100:200] = 1

#         # Threshold the map
#         _, thresholded_map = cv2.threshold(map_array, 0.9, 1, cv2.THRESH_BINARY)

#         # Dilate the obstacles on the map
#         dilated_map =   cv2.dilate(
#                             src = thresholded_map, 
#                             kernel = np.ones((31,31), np.uint8), 
#                             iterations = 1
#                         )
        
#         (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(dilated_map, connectivity=4)

#         polygons = []
#         for i in range(numLabels):
#             x = stats[i, cv2.CC_STAT_LEFT]
#             y = stats[i, cv2.CC_STAT_TOP]
#             w = stats[i, cv2.CC_STAT_WIDTH]
#             h = stats[i, cv2.CC_STAT_HEIGHT]
#             polygons.append([(x,y), (x+w,y), (x+w,y+h), (x,y+h)])    
#     print(f"Opencv connect time: {round(time.time()-start,4)}s")

#     start = time.time()
#     for _ in range(1000):
#         map_array = np.zeros((500, 300)).astype(np.uint8)
#         map_array[20:40, 30:60] = 1
#         map_array[100:200, 100:200] = 1
#         map_array[300:400, 100:200] = 1

#         # Threshold the map
#         _, thresholded_map = cv2.threshold(map_array, 0.9, 1, cv2.THRESH_BINARY)

#         # Dilate the obstacles on the map
#         dilated_map =   cv2.dilate(
#                             src = thresholded_map, 
#                             kernel = np.ones((31,31), np.uint8), 
#                             iterations = 1
#                         )
        
#         # Extract the contours of the obstacles from the map using opencv
#         contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Extract the polygons from the contours using opencv
#         polygons = []
#         for contour in contours:
#             contour_max = np.max(contour, axis=(0,1))
#             contour_min = np.min(contour, axis=(0,1))
#             polygons.append([(contour_min[0], contour_min[1]), (contour_max[0], contour_min[1]), 
#                              (contour_max[0], contour_max[1]), (contour_min[0], contour_max[1])])
#     print(f"Opencv contours time: {round(time.time()-start,4)}s")


# if __name__ == "__main__":
#     test_map()


    # def _findPolygons(self):
    #     # copy the map and transpose it (because cv2 takes y-axis in 1. and x-axes in 2. dimension)
    #     map_array = self._map.copy().astype(np.float32).transpose()

    #     # Threshold the map
    #     _, thresholded_map = cv2.threshold(map_array, self._threshold, 1, cv2.THRESH_BINARY)

    #     # Dilate the obstacles on the map
    #     dilated_map =   cv2.dilate(
    #                         src = thresholded_map, 
    #                         kernel = np.ones((self._kernel_size, self._kernel_size), np.uint8), 
    #                         iterations = 1
    #                     )
        
    #     # Extract the contours of the obstacles from the map using opencv
    #     contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     # Extract the polygons from the contours using opencv
    #     polygons = []
    #     for contour in contours:
    #         # Approximate the contour with a polygon
    #         polygon = cv2.approxPolyDP(contour, self._contour_approx * cv2.arcLength(contour, True), True)
    #         # Convert the polygon to a list of points
    #         polygon_points = [tuple(point[0]) for point in polygon]
    #         # Add the polygon to the list of polygons
    #         polygons.append(polygon_points)

    #     return polygons, dilated_map

    # def _moveOutPolygon(self, polygons, start):
    #     # find polygon that contains start
    #     idx = self.vis.insidePolygon(polygons=polygons, point=start)     
        
    #     # calculate mean of closest polygon
    #     x_mean = np.mean([p[0] for p in polygons[idx]], dtype=np.uint32)
    #     y_mean = np.mean([p[1] for p in polygons[idx]], dtype=np.uint32)

    #     # set goal that drone moves away from polygon
    #     goal = (np.clip(2*start[0]-x_mean, 0, self._map.shape[0]), np.clip(2*start[1]-y_mean, 0, self._map.shape[1]))
    #     path = [start, goal]

    #     # set inside polygon index for drawing
    #     self._start_in_polygon = idx
    #     return path


#  def drawMap(self, sensor_data, real_command, desired_command):
#         if not self.visualization:
#             return
        

#         polygons = copy.deepcopy(self._polygons)
#         path = copy.copy(self._path)

#         # create image and copy map
#         map_img = np.ones(self.img.shape, dtype=np.uint8) * 255
#         map_array = np.transpose(self._map).copy()

#         # # draw map_array in grayscale
#         map_array = np.clip(map_array, 0, 1)
#         map_img[:, :, 0] = (1-map_array) * 255
#         map_img[:, :, 1] = (1-map_array) * 255
#         map_img[:, :, 2] = (1-map_array) * 255
        
#         # draw polygons in green (or orange if current position is in polygon)
#         for idx, poly in enumerate(polygons):
#             for i in range(len(poly)):
#                 if i == len(poly)-1:
#                     rr, cc = self.skimage_draw_line(poly[i][0], poly[i][1], poly[0][0], poly[0][1])        
#                 else:
#                     rr, cc = self.skimage_draw_line(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
#                 rr = np.clip(rr, 0, self._map.shape[0]-1)
#                 cc = np.clip(cc, 0, self._map.shape[1]-1)
#                 if self._start_in_polygon == idx: # orange
#                     map_img[cc, rr, 0] = 255
#                     map_img[cc, rr, 1] = 165
#                     map_img[cc, rr, 2] = 0
#                 else: # green
#                     map_img[cc, rr, 0] = 0
#                     map_img[cc, rr, 1] = 255
#                     map_img[cc, rr, 2] = 0

#         # draw polygon center in orange if current position is inside polygon
#         if self._start_in_polygon is not None:  
#             x_mean = np.mean([p[0] for p in polygons[self._start_in_polygon]], dtype=np.uint32)
#             y_mean = np.mean([p[1] for p in polygons[self._start_in_polygon]], dtype=np.uint32)
#             rr, cc = self.skimage_draw_ellipse(x_mean, y_mean, r_radius=4, c_radius=4, shape=self._map.shape)
#             map_img[cc, rr, 0] = 255
#             map_img[cc, rr, 1] = 165
#             map_img[cc, rr, 2] = 0

#         # draw path in violett
#         for i in range(len(path)-1):
#             # print(f"_drawMap: path[i]: {path[i]}, path[i+1]: {path[i+1]}")
#             rr, cc = self.skimage_draw_line(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
#             rr = np.clip(rr, 0, self._map.shape[0]-1) 
#             cc = np.clip(cc, 0, self._map.shape[1]-1)
#             map_img[cc, rr, 0] = 238
#             map_img[cc, rr, 1] = 130
#             map_img[cc, rr, 2] = 238       

#         # draw drone position and real velocity in blue
#         x0 = self._pos2idx(sensor_data['x_global'], "x")
#         y0 = self._pos2idx(sensor_data['y_global'], "y")
#         x1 = self._pos2idx(sensor_data['x_global'] + real_command[0], "x")
#         y1 = self._pos2idx(sensor_data['y_global'] + real_command[1], "y")
#         rr_line, cc_line = self.skimage_draw_line(x0, y0, x1, y1)
#         rr_circle, cc_circle, = self.skimage_draw_ellipse(x0, y0, r_radius=4, c_radius=4, shape=self._map.shape)
#         rr = np.concatenate((rr_line, rr_circle))
#         cc = np.concatenate((cc_line, cc_circle))
#         map_img[cc, rr, 0] = 0
#         map_img[cc, rr, 1] = 0
#         map_img[cc, rr, 2] = 255

#         # draw desired velocity in red
#         x1 = self._pos2idx(sensor_data['x_global'] + desired_command[0], "x")
#         y1 = self._pos2idx(sensor_data['y_global'] + desired_command[1], "y")
#         rr, cc = self.skimage_draw_line(x0, y0, x1, y1)
#         map_img[cc, rr, 0] = 255
#         map_img[cc, rr, 1] = 0
#         map_img[cc, rr, 2] = 0

#         # save img
#         self.img[:] = map_img[:]

#         # trigger plotting event
#         self.event.set()