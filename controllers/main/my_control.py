# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np
import copy
import time

from navigation import Navigation
from parameters import Parameters

"""
The VISUALIZATION class is only needed to visualize the drone in the simulation.
This import may be ignored if you don't need any visualization.
"""
try:
    from visualization import Visualization
except:
    pass



# Don't change the class name of 'MyController'
class MyController():
    def __init__(self):
        self.params = Parameters()
        self.nav = Navigation(params=self.params)
        if self.params.vis:
            self.vis = Visualization(nav=self.nav, params=self.params)

        """
        State machine:
            - takeoff
            - search (search for landing or starting platform)
            - land
            - reset (return to starting platform or stop)
        """
        self._state = "takeoff"

        """
        Two parts:
            True: drone goes from starting area to landing platform
            False: drone goes from landing area to starting platform
        """
        self._first_part = True 
        self._search_points = [(3.9,1.5)] # list of points to visit
        self._search_points_idx = 0 # index of the current point
        self._init_position = None # initial position of the drone when taking off

        self._real_theta = 0.0 # real theta of the drone
        self._real_speed = 0.0 # real speed of the drone
        self._real_height = 0.0

        self._platform_points = [] # list of points of the platform
        self._scan_points = [] # list of points of the scan
        self._scan_points_idx = 0 # index of the current scan point
        self._desired_height = self.params.mc_height_search

        self._land_pos =(0.0, 0.0)
        self._land_vel = (0.0, 0.0)
        self._land_count = 0
        self._yaw_approx = 0.0


    # Don't change the method name of 'step_control'
    def step_control(self, sensor_data):
        # measure time of entire control loop
        step_start_time = time.time()

        # global path planner: state machine
        desired_command = [0.0, 0.0, 0.0, 0.0]
        if self._state == "takeoff":
            real_command = self._takeoff(sensor_data)
        elif self._state == "search":                 
            real_command, desired_command = self._search(sensor_data)
        elif self._state == "scan":
            real_command, desired_command = self._scan(sensor_data)       
        elif self._state == "land":
            real_command = self._land(sensor_data)
        elif self._state == "reset":
            real_command = self._reset()
        else:
            raise Exception("Invalid state")
        
        # draw map to image
        if self.params.vis:
            self.vis.drawMap(sensor_data=sensor_data, real_command=real_command, desired_command=desired_command)

        # convert command from global to local frame
        local_command = self._gloabal2local(sensor_data=sensor_data, command=real_command)

        # sleep to ensure no rush
        step_time = time.time() - step_start_time
        if step_time < self.params.mc_min_control_loop_time:
            time.sleep(self.params.mc_min_control_loop_time - step_time)

        # convert command from global to local frame
        return local_command
        

    def _takeoff(self, sensor_data):
        if sensor_data['range_down'] < self._desired_height-0.01:
            self._real_height += 0.01 * (self._desired_height-sensor_data['range_down'])/self._desired_height
            command = [0.0, 0.0, 0.0, self._real_height]
        else:
            command = [0.0, 0.0, 0.0, self._desired_height]
            self._state = "search"
            if self.params.verb: print("_takeoff: takeoff -> search")

            if self._init_position is None and self._first_part:
                self._init_position = (sensor_data["x_global"], sensor_data["y_global"])
        
        return command
    
    def _search(self, sensor_data):
        # Update the map
        self.nav.updateMap(sensor_data=sensor_data)

        # check if drone is over platform     
        if self._platformTransition(sensor_data=sensor_data):
            # transition to landing
            self._state = "land"
            if self.params.verb: print("_search: search -> land")

            # calculate landing velocity
            # theta = np.arctan2(sensor_data["y_global"]-self._last_pos[1], sensor_data["x_global"]-self.last_pos[0])
            theta = self._real_theta #self._yaw_approx #sensor_data['yaw'] #
            self._land_vel = (np.cos(theta)*self.params.mc_land_speed, np.sin(theta)*self.params.mc_land_speed)
            self._land_pos = (sensor_data["x_global"], sensor_data["y_global"])
            self

            # stop moving
            command = [self._land_vel[0], self._land_vel[1], 0.0, self.params.mc_height_platform]
            return command, command
            
        
        # remember last position to calculate landing velocity
        if sensor_data['yaw'] - self._yaw_approx > np.pi:
            self._yaw_approx += 2*np.pi
        elif sensor_data['yaw'] - self._yaw_approx < -np.pi:
            self._yaw_approx -= 2*np.pi
        self._yaw_approx = sensor_data['yaw']*0.5 + self._yaw_approx*0.5
        self.last_pos = (sensor_data["x_global"], sensor_data["y_global"])

        # find shortest path to next point
        desired_theta, goal_in_polygon = self.nav.findPath(sensor_data=sensor_data, goal=self._search_points[self._search_points_idx]) 
        if goal_in_polygon:
            desired_theta = self._real_theta

        # check increase point index if next set point is reached
        self._checkPoint(sensor_data=sensor_data, dist=0.05, goal_in_polygon=goal_in_polygon)        

        # return desired and real command
        return self._theta2command(desired_theta, max_speed=self.params.mc_max_speed)
    
    # def _scan(self, sensor_data):
    #     # Update the map
    #     self.nav.updateMap(sensor_data=sensor_data)

    #     # add point to platform points if drone is over platform
    #     if self._platformTransition(sensor_data=sensor_data):
    #         self._platform_points.append((sensor_data["x_global"], sensor_data["y_global"]))

    #     # define goal
    #     if len(self._platform_points) >= self.params.mc_min_nb_platform_points and self._scan_points_idx%3 == 2:
    #         # define goal as mean of all platform points
    #         goal = (np.mean(np.array(self._platform_points)[:,0]), np.mean(np.array(self._platform_points)[:,1]))           
    #     else:
    #         goal = self._scan_points[self._scan_points_idx]

    #     # check if goal is reached  
    #     if self._dist2point(sensor_data=sensor_data, point=goal) < self.params.mc_min_dist_platform_reached:
    #         # transition to landing if mean platform position is reached or all scan points are checked
    #         if (len(self._platform_points) >= self.params.mc_min_nb_platform_points and self._scan_points_idx%3 == 2)\
    #             or (self._scan_points_idx >= len(self._scan_points)):
    #             self._state = "land"
    #             if self.params.verb: print("_search: scan -> land")
    #             command = [0.0, 0.0, 0.0, self._desired_height]
    #             return command, command
    #         else:
    #             # increase scan point index
    #             self._scan_points_idx += 1

    #     desired_theta, goal_in_polygon = self.nav.findPath(sensor_data=sensor_data, goal=goal)
    #     real_command, desired_command = self._theta2command(desired_theta, max_speed=self.params.mc_land_speed)

    #     return real_command, desired_command
    
    def _land(self, sensor_data):
        # # continue moveing in same direction for some time
        # self._land_count += 1
        # if self._land_count < self.params.mc_land_count_max:
        #     return [self._land_vel[0], self._land_vel[1], 0.0, self.params.mc_height_platform]
        
        if self._dist2point(sensor_data=sensor_data, point=self._land_pos) < self.params.mc_land_dist:
            return [self._land_vel[0], self._land_vel[1], 0.0, self.params.mc_height_platform]
        
        # land
        if sensor_data['range_down'] > 0.02:
            self._real_height -= 0.01 * (sensor_data['range_down'])/self._desired_height
        else:
            self._state = "reset"
            if self.params.verb: print("_land: land -> reset")
        
        return [0.0, 0.0, 0.0, self._real_height]
    
    def _reset(self):
        # return to starting area
        if self._first_part:
            self._first_part = False
            self._search_points = [self._init_position]
            self._search_points_idx = 0
            self._land_count = 0

            self._state = "takeoff"
            if self.params.verb: print("_reset: reset -> takeoff")

        return [0.0, 0.0, 0.0, self._real_height]
    
    def _theta2command(self, desired_theta, max_speed):
        # if desired theta is None (no path was found), use last theta
        if desired_theta is None:
            desired_theta = self._real_theta
            print(f"_theta2command: desired_theta is None, using last theta: {desired_theta:.3f}")

        # shift desired theta s.t. it is close (|error| <= pi) to real theta
        theta_error = desired_theta - self._real_theta
        if theta_error > np.pi:
            desired_theta -= 2*np.pi
        elif theta_error <= -np.pi:
            desired_theta += 2*np.pi
        
        # update real theta with desired theta
        self._real_theta = (1-self.params.mc_gamma_theta)*self._real_theta + self.params.mc_gamma_theta*desired_theta

        # calculate real speed: max if real and desired theta are aligned, min if |error| >= pi/2
        desired_speed = (1-abs(desired_theta-self._real_theta)/(np.pi/2)) * max_speed
        desired_speed = np.clip(desired_speed, self.params.mc_min_speed, max_speed)
        self._real_speed = (1-self.params.mc_gamma_speed)*self._real_speed + self.params.mc_gamma_speed*desired_speed
        
        # normalize real theta to (-pi, pi]
        if self._real_theta > np.pi:
            self._real_theta -= 2*np.pi
        elif self._real_theta <= -np.pi:
            self._real_theta += 2*np.pi

        # return real and desired commands
        real_command = [self._real_speed*np.cos(self._real_theta), self._real_speed*np.sin(self._real_theta), 
                        self.params.mc_explore_speed, self._desired_height]
        desired_command = [np.cos(desired_theta)*self.params.mc_max_speed, np.sin(desired_theta)*self.params.mc_max_speed, 
                           self.params.mc_explore_speed, self._desired_height]     
        return real_command, desired_command

    
    def _platformTransition(self, sensor_data):
        # check if drone is in landing or starting region
        if ((self._first_part and sensor_data['x_global']<self.params.map_landing_region_x[0]) \
            or (not self._first_part and sensor_data['x_global']>self.params.map_starting_region_x[1])):
            return False
        
        if self._desired_height-sensor_data['range_down'] > self.params.mc_height_delta:
            print("detect platform transition")
            self._desired_height = sensor_data['range_down']
            return True
        
        return False

        # if self._over_platform and (sensor_data['range_down']-self._desired_height > self.params.mc_height_delta):
        #     # detect transition from platform to search area
        #     self._over_platform = False
        #     self._desired_height = self.params.mc_height_search
        #     if self.params.verb: print("_platformTransition: over platform -> search area")
        # elif not self._over_platform and (self._desired_height-sensor_data['range_down'] > self.params.mc_height_delta):
        #     # detect transition from search area to platform
        #     self._over_platform = True
        #     self._desired_height = self.params.mc_height_platform
        #     if self.params.verb: print("_platformTransition: search area -> over platform")
        #     return True

        # return False
        
    def _init_scan_points(self, sensor_data):
        # rotational matrix in direction of movement
        cos_theta = np.cos(self._real_theta)
        sin_theta = np.sin(self._real_theta)
        rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # scanning displacement
        dx = self.params.mc_scan_dist
        dy = self.params.mc_scan_dist
        delta_dist = np.array([ (dx,0), (dx,dy), (0,dy),
                                (0,-dy), (-dx,-dy), (-dx,0),
                                (2*dx,0), (2*dx,-2*dy), (0,-2*dy),
                                (0,2*dy), (-2*dx,2*dy), (-2*dx,0),
                                (3*dx,0), (3*dx,3*dy), (0,3*dy),
                                (0,-3*dy), (-3*dx,-3*dy), (-3*dx,0),
                                (4*dx,0), (4*dx,-4*dy), (0,-4*dy),
                                (0,4*dy), (-4*dx,4*dy), (-4*dx,0),])
        delta_dist = np.transpose(np.matmul(rot_matrix, np.transpose(delta_dist)))

        # scanning center point
        x = sensor_data['x_global']
        y = sensor_data['y_global']
        pos = np.tile(np.array([x, y]), (delta_dist.shape[0], 1))

        self._platform_points = []
        self._scan_points_idx = 0
        self._scan_points = pos + delta_dist
        self._scan_points[:,0] = np.clip(self._scan_points[:,0], 
                                         self.params.map_size_x[0] + 1.2*self.params.map_boarder_size*self.params.map_res, 
                                         self.params.map_size_x[1] - 1.2*self.params.map_boarder_size*self.params.map_res)
        self._scan_points[:,1] = np.clip(self._scan_points[:,1], 
                                         self.params.map_size_y[0] + 1.2*self.params.map_boarder_size*self.params.map_res, 
                                         self.params.map_size_y[1] - 1.2*self.params.map_boarder_size*self.params.map_res)
    
    def _gloabal2local(self, sensor_data, command):
        vx = command[0]*np.cos(sensor_data['yaw']) + command[1]*np.sin(sensor_data['yaw'])
        vy = -command[0]*np.sin(sensor_data['yaw']) + command[1]*np.cos(sensor_data['yaw'])
        return [vx, vy, command[2], command[3]]
    
    def _dist2point(self, sensor_data, point):
        v_x = point[0] - sensor_data['x_global']
        v_y = point[1] - sensor_data['y_global']
        return np.sqrt(v_x**2 + v_y**2)
    
    def _checkPoint(self, sensor_data, dist, goal_in_polygon):
        # increase point index if goal is in polygon
        if goal_in_polygon is not None:
            self._incPointIndex()

            if self.params.verb:
                print(f"_checkPoint: Goal in polygon: {goal_in_polygon}")
            return False

        # increase point index if drone is close enough to the point
        if self._dist2point(sensor_data, self._search_points[self._search_points_idx]) < dist:        
            self._incPointIndex()

            if self.params.verb:
                print(f"_checkPoint: idx: {self._search_points_idx}, point: {self._search_points[self._search_points_idx]} reached")
            return True

        return False
    
    def _incPointIndex(self):
        # increase point index
        self._search_points_idx += 1

        # add search points if index is out of range
        if self._search_points_idx >= len(self._search_points):
            self._search_points_idx = 0
            self._search_points = []
            if self._first_part:
                # count the number of polygons in the lower and upper part of the landing region
                polygons = self.nav.get('unfiltered_polygons')
                upper_counter = 0
                lower_counter = 0
                for poly in polygons:
                    poly_mean = np.mean(np.array(poly), axis=0)

                    # ignore polygons outside of the landing region
                    if poly_mean[0] < self.params.map_landing_region_x[0] / self.params.map_res:
                        continue

                    if poly_mean[1] > ((self.params.map_size_y[1]-self.params.map_size_y[0])/2) / self.params.map_res:
                        upper_counter += 1
                    else:
                        lower_counter += 1
                
                if self.params.verb:
                    print(f"_incPointIndex: upper_counter: {upper_counter}, lower_counter: {lower_counter}")

                # search first the part with fewer polygons
                if upper_counter > lower_counter:
                    self._search_points = self._search_points + self.params.mc_search_points_lower + self.params.mc_search_points_upper
                else:
                    self._search_points = self._search_points + self.params.mc_search_points_upper + self.params.mc_search_points_lower
            else:
                self._search_points = [self._init_position]

    # def _init_search_points(self, sensor_data):
    #     # add searching points from landing region  
    #     self._points = copy.deepcopy(self.params.mc_first_part_points)

    #     # add starting point
    #     x = sensor_data['x_global']
    #     y = sensor_data['y_global']
    #     self._points.append((x, y))
    #     self._second_part_points_idx = len(self._points)-1

    #     # # add searching points from second part
    #     # self._points = self._points + [(x, y+0.1), (x+0.1, y+0.1), (x+0.1, y-0.1), (x-0.1, y-0.1), 
    #     #                                 (x-0.1, y+0.2), (x+0.2, y+0.2), (x+0.2, y-0.2), (x-0.2, y-0.2),
    #     #                                 (x-0.2, y+0.3), (x+0.3, y+0.3), (x+0.3, y-0.3), (x-0.3, y-0.3),
    #     #                                 (x-0.3, y+0.4), (x+0.4, y+0.4), (x+0.4, y-0.4), (x-0.4, y-0.4),
    #     #                                 (x-0.4, y+0.5), (x+0.5, y+0.5), (x+0.5, y-0.5), (x-0.5, y-0.5),
    #     #                                 (x-0.5, y+0.6), (x+0.6, y+0.6), (x+0.6, y-0.6), (x-0.6, y-0.6),
    #     #                                 (x-0.6, y+0.7), (x+0.7, y+0.7), (x+0.7, y-0.7), (x-0.7, y-0.7),
    #     #                                 (x-0.7, y+0.8), (x+0.8, y+0.8), (x+0.8, y-0.8), (x-0.8, y-0.8),
    #     #                                 (x-0.8, y+0.9), (x+0.9, y+0.9), (x+0.9, y-0.9), (x-0.9, y-0.9),
    #     #                                 (x-0.9, y+1.0), (x+1.0, y+1.0), (x+1.0, y-1.0), (x-1.0, y-1.0),
    #     #                                 (x-1.0, y+1.1), (x+1.1, y+1.1), (x+1.1, y-1.1), (x-1.1, y-1.1),
    #     #                                 (x-1.1, y+1.2), (x+1.2, y+1.2), (x+1.2, y-1.2), (x-1.2, y-1.2),
    #     #                                 (x-1.2, y+1.3), (x+1.3, y+1.3), (x+1.3, y-1.3), (x-1.3, y-1.3),
    #     #                                 (x-1.3, y+1.4), (x+1.4, y+1.4), (x+1.4, y-1.4), (x-1.4, y-1.4),
    #     #                                 (x-1.4, y+1.5), (x+1.5, y+1.5), (x+1.5, y-1.5), (x-1.5, y-1.5)]
    
    # def _move(self, sensor_data, next_point):
    #     # command = [0.0, 0.0, 0.0, self._height_desired]
    #     # self._state = "search"
    #     # print("_move: move -> search")
    #     # return command

    #     command = self._move2point(sensor_data, next_point)

    #     # check if point is reached and increase point index if this is the case
    #     self._checkPoint(sensor_data=sensor_data, dist=0.05)

    #     # start searching if landing or starting area is reached
    #     if (self._first_part and sensor_data['x_global']>self._landing_region_x[0]) \
    #         or (not self._first_part and sensor_data['x_global']<self._starting_region_x[1]):
    #         command = [0.0, 0.0, 0.0, self._height_desired]
    #         self._state = "search"

    #         if self._verb:
    #             print("_move: move -> search")

    #     # self._checkForObstacles(sensor_data=sensor_data, command=command)

    #     return self._scaleCommand(command=command, scale=0.2)
    
    # def _avoid(self, sensor_data):

    #     # control command
    #     sensors = np.array([sensor_data["range_front"], sensor_data["range_left"], 
    #                         sensor_data["range_back"], sensor_data["range_right"]])
    #     critical = (sensors < 0.5) & (sensors <= np.min(sensors) + 0.001)

    #     # return to state before obstacle avoidance if obstacle was surrounded
    #     if not (critical == self._avoid_critical).all():
    #         self._avoid_counter += 1

    #     if self._avoid_counter > 100:
    #         self._avoid_counter = 0
    #         self._state = self._state_before_avoid

    #         if self._verb:
    #             print(f"_avoid: avoid -> {self._state}")
    #         return [0.0, 0.0, 0.0, 0.5]
        
    #     # check if landing platform is reached
    #     if sensor_data['range_down'] < 0.42 and self._state_before_avoid == "search":
    #         self._state = "land"

    #         if self._verb:
    #             print("_search: search -> land")
    #         return [0.0, 0.0, 0.0, self._height_desired]

    #     # skip next point if it is too close to the obstacle
    #     if self._checkPoint(sensor_data=sensor_data, dist=0.75) and self._verb:
    #         print("_avoid: skip point")

    #     # determine direction to surround obstacles
    #     command = [0.0, 0.0, 0.0, 0.5]
    #     if critical[0] or critical[2]:
    #         if critical[0]:
    #             command[0] = -0.05
    #         else:
    #             command[0] = 0.05
    #         command[1] = 0.2 * self._avoid_direction

    #     elif critical[1] or critical[3]:
    #         if critical[1]:
    #             command[1] = -0.05
    #         else:
    #             command[1] = 0.05
    #         command[0] = 0.2 * self._avoid_direction


    #     return self._scaleCommand(command=command, scale=0.2)      

    # def _checkForObstacles(self, sensor_data, command):
    #     sensors = np.array([sensor_data["range_front"], sensor_data["range_left"], 
    #                         sensor_data["range_back"], sensor_data["range_right"]])
    #     critical = (sensors < 0.4) & (sensors <= np.min(sensors) + 0.0001)
    #     critical2 = (sensors < 0.5) & np.invert(critical)

    #     if np.any(critical): 
    #         self._state_before_avoid = self._state
    #         self._state = "avoid"
    #         self._avoid_critical = critical

    #         if critical[0] or critical[2]:
    #             # move towards next point while avoiding obstacle
    #             if self._points[self._points_idx][1] >= sensor_data['y_global']:
    #                 self._avoid_direction = 1
    #             else:
    #                 self._avoid_direction = -1

    #             # move not ouside arena while avoiding obstacle
    #             if np.abs(sensor_data['y_global']-self._landing_region_y[0]) < 1.0:
    #                 self._avoid_direction = 1
    #             elif np.abs(self._landing_region_y[1]-sensor_data['y_global']) < 1.0:
    #                 self._avoid_direction = -1
                
    #             # move away from second obstacle while avoiding first obstacle
    #             if critical2[1]:
    #                 self._avoid_direction = -1
    #             elif critical2[3]:
    #                 self._avoid_direction = 1

    #         elif critical[1] or critical[3]:
    #             # move towards next point while avoiding obstacle
    #             if self._points[self._points_idx][0] >= sensor_data['x_global']:
    #                 self._avoid_direction = 1
    #             else:
    #                 self._avoid_direction = -1

    #             # move not ouside arena while avoiding obstacle
    #             if np.abs(sensor_data['x_global']-self._starting_region_x[0]) < 1.0:
    #                 self._avoid_direction = 1
    #             elif np.abs(self._landing_region_x[1]-sensor_data['x_global']) < 1.0:
    #                 self._avoid_direction = -1

    #             # move away from second obstacle while avoiding first obstacle
    #             if critical2[0]:
    #                 self._avoid_direction = -1
    #             elif critical2[2]:
    #                 self._avoid_direction = 1

    #         if self._verb:
    #             print(f"_checkForObstacles: {self._state_before_avoid} -> avoid")
        

