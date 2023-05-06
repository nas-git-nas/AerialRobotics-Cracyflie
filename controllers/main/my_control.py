# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np
import copy

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

        self._points = [] # list of points to visit
        self._points_idx = 0 # index of the current point
        self._second_part_points_idx = None # index where the points from the second part start

        self._real_theta = 0.0 # real theta of the drone
        self._real_speed = 0.0 # real speed of the drone
        self._real_height = 0.0

        self._platform_points = [] # list of points of the platform
        self._scan_radius = None
        self._scan_angle = None


    # Don't change the method name of 'step_control'
    def step_control(self, sensor_data):
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
        return self._gloabal2local(sensor_data=sensor_data, command=real_command)
        

    def _takeoff(self, sensor_data):
        if sensor_data['range_down'] < self.params.mc_height_search-0.01:
            self._real_height += 0.005
            command = [0.0, 0.0, 0.0, self._real_height]
        else:
            command = [0.0, 0.0, 0.0, self.params.mc_height_search]
            self._state = "search"
            if self.params.verb: print("_takeoff: takeoff -> search")

            # initialize points if not already done
            if len(self._points) == 0:
                self._init_points(sensor_data=sensor_data)
        
        return command
    
    def _search(self, sensor_data):
        # Update the map
        self.nav.updateMap(sensor_data=sensor_data)

        # find shortest path to next point
        desired_theta, goal_in_polygon = self.nav.findPath(sensor_data=sensor_data, goal=self._points[self._points_idx])  

        # check increase point index if next set point is reached
        self._checkPoint(sensor_data=sensor_data, dist=0.05, goal_in_polygon=goal_in_polygon)    

        # check if drone is in landing or starting area
        if ((self._first_part and sensor_data['x_global']>self.params.map_landing_region_x[0]) \
            or (not self._first_part and sensor_data['x_global']<self.params.map_starting_region_x[1])):
            # check if landing platform is reached
            if sensor_data['range_down'] < 0.42:
                # transition to landing
                self._state = "scan"
                if self.params.verb: print("_search: search -> scan")

                # set spiral parameters for the scanning
                self._platform_points = [(sensor_data["x_global"], sensor_data["y_global"])]
                self._scan_angle = desired_theta
                self._scan_radius = self.params.mc_scan_delta_radius

        real_command, desired_command = self._theta2command(desired_theta, max_speed=self.params.mc_max_speed)
        return real_command, desired_command
    
    def _scan(self, sensor_data):
        # Update the map
        self.nav.updateMap(sensor_data=sensor_data)

        # set goal and calculate command
        if len(self._platform_points) < self.params.mc_min_nb_platform_points:
            # define goal as spiral around the point where the platform was detected
            x = self._platform_points[-1][0] + np.cos(self._scan_angle)*self._scan_radius
            y = self._platform_points[-1][1] + np.sin(self._scan_angle)*self._scan_radius
            goal = (np.clip(x, self.params.map_size_x[0] + 2*self.params.map_boarder_size*self.params.map_res, 
                            self.params.map_size_x[1] - 2*self.params.map_boarder_size*self.params.map_res),
                    np.clip(y, self.params.map_size_y[0] + 2*self.params.map_boarder_size*self.params.map_res, 
                            self.params.map_size_y[1] - 2*self.params.map_boarder_size*self.params.map_res))
        else:
            # define goal as mean of all platform points
            platform_points = np.array(self._platform_points)
            goal = (np.mean(platform_points[:,0]), np.mean(platform_points[:,1]))

            # transition to landing if mean platform position is reached
            if self._dist2point(sensor_data=sensor_data, point=goal) < self.params.mc_min_dist_platform_reached:
                self._state = "land"
                if self.params.verb: print("_search: scan -> land")
                command = [0.0, 0.0, 0.0, self.params.mc_height_scan]
                return command, command

        desired_theta, goal_in_polygon = self.nav.findPath(sensor_data=sensor_data, goal=goal)
        real_command, desired_command = self._theta2command(desired_theta, max_speed=self.params.mc_land_speed)

        # remember position if it is over landing platform
        if self.params.mc_height_search - sensor_data["range_down"] > self.params.mc_height_delta:
            self._platform_points.append((sensor_data["x_global"], sensor_data["y_global"]))
            real_command[3] = self.params.mc_height_scan

        # increase scanning radius and angle
        if self._dist2point(sensor_data=sensor_data, point=goal) < self.params.mc_min_dist_platform_reached \
            or goal_in_polygon:
            self._scan_angle += self.params.mc_scan_delta_angle
            self._scan_radius += self.params.mc_scan_delta_radius

        return real_command, desired_command
    
    def _land(self, sensor_data):
        if sensor_data['range_down'] > 0.02:
            self._real_height -= 0.005
        else:
            self._state = "reset"
            if self.params.verb: print("_land: land -> reset")
        
        return [0.0,  0.0, 0.0, self._real_height]
    
    def _reset(self):
        # return to starting area
        if self._first_part:
            self._first_part = False
            self._points_idx = self._second_part_points_idx

            self._state = "takeoff"
            if self.params.verb: print("_reset: reset -> takeoff")

        return [0.0, 0.0, 0.0, 0.0]
    
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
                        self.params.mc_explore_speed, self.params.mc_height_search]
        desired_command = [np.cos(desired_theta)*self.params.mc_max_speed, np.sin(desired_theta)*self.params.mc_max_speed, 
                           self.params.mc_explore_speed, self.params.mc_height_search]     
        return real_command, desired_command
    
    def _init_points(self, sensor_data):
        # add searching points from landing region  
        self._points = copy.deepcopy(self.params.mc_first_part_points)

        # add starting point
        x = sensor_data['x_global']
        y = sensor_data['y_global']
        self._points.append((x, y))
        self._second_part_points_idx = len(self._points)-1

        # add searching points from second part
        self._points = self._points + [(x, y+0.1), (x+0.1, y+0.1), (x+0.1, y-0.1), (x-0.1, y-0.1), 
                                        (x-0.1, y+0.2), (x+0.2, y+0.2), (x+0.2, y-0.2), (x-0.2, y-0.2),
                                        (x-0.2, y+0.3), (x+0.3, y+0.3), (x+0.3, y-0.3), (x-0.3, y-0.3),
                                        (x-0.3, y+0.4), (x+0.4, y+0.4), (x+0.4, y-0.4), (x-0.4, y-0.4),
                                        (x-0.4, y+0.5), (x+0.5, y+0.5), (x+0.5, y-0.5), (x-0.5, y-0.5),
                                        (x-0.5, y+0.6), (x+0.6, y+0.6), (x+0.6, y-0.6), (x-0.6, y-0.6),
                                        (x-0.6, y+0.7), (x+0.7, y+0.7), (x+0.7, y-0.7), (x-0.7, y-0.7),
                                        (x-0.7, y+0.8), (x+0.8, y+0.8), (x+0.8, y-0.8), (x-0.8, y-0.8),
                                        (x-0.8, y+0.9), (x+0.9, y+0.9), (x+0.9, y-0.9), (x-0.9, y-0.9),
                                        (x-0.9, y+1.0), (x+1.0, y+1.0), (x+1.0, y-1.0), (x-1.0, y-1.0),
                                        (x-1.0, y+1.1), (x+1.1, y+1.1), (x+1.1, y-1.1), (x-1.1, y-1.1),
                                        (x-1.1, y+1.2), (x+1.2, y+1.2), (x+1.2, y-1.2), (x-1.2, y-1.2),
                                        (x-1.2, y+1.3), (x+1.3, y+1.3), (x+1.3, y-1.3), (x-1.3, y-1.3),
                                        (x-1.3, y+1.4), (x+1.4, y+1.4), (x+1.4, y-1.4), (x-1.4, y-1.4),
                                        (x-1.4, y+1.5), (x+1.5, y+1.5), (x+1.5, y-1.5), (x-1.5, y-1.5)]
    
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
        if self._dist2point(sensor_data, self._points[self._points_idx]) < dist:        
            self._incPointIndex()

            if self.params.verb:
                print(f"_checkPoint: idx: {self._points_idx}, point: {self._points[self._points_idx]} reached")
            return True

        return False
    
    def _incPointIndex(self):
        # increase point index
        self._points_idx += 1

        # reset point index if it is out of range
        if self._first_part:           
            if self._points_idx >= self._second_part_points_idx:
                self._points_idx = 0
        else:
            if self._points_idx >= len(self._points):
                self._points_idx = self._second_part_points_idx
    
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
        

