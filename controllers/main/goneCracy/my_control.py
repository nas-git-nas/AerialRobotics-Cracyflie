# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np

from occupancy_map import OccupancyMap



# Don't change the class name of 'MyController'
class MyController():
    def __init__(self):
        self._verb = True
        self.visualization = False

        self._height_desired = 0.5
        self._height_current = 0.0

        self._landing_region_x = (3.5, 5) # x range of landing region
        self._landing_region_y = (0, 3) # y range of landing region
        self._starting_region_x = (0, 1.5) # x range of starting region
        self._starting_region_y = (0, 3) # y range of starting region

        """
        State machine:
            - takeoff
            - move (move to landing or starting area)
            - search (search for landing or starting pad)
            - land
            - stop
            - avoid
        """
        self._state = "takeoff"

        """
        Two parts:
            True: drone goes from starting area to landing platform
            False: drone goes from landing area to starting platform
        """
        self._first_part = True 

        self._points = [(3.70, 1.40), (3.70, 1.00), (3.70, 0.60), (3.70, 0.20),
                        (4.25, 0.20), (4.80, 0.20), (4.80, 0.50), (4.40, 0.50), (4.00, 0.50),
                        (4.00, 0.80), (4.40, 0.80), (4.80, 0.80), (4.80, 1.10), (4.40, 1.10), (4.00, 1.10),
                        (4.00, 1.40), (4.40, 1.40), (4.80, 1.40), (4.80, 1.70), (4.40, 1.70), (4.00, 1.70), (3.70, 1.70),
                        (3.70, 2.00), (4.00, 2.00), (4.40, 2.00), (4.80, 2.00), (4.80, 2.30), (4.40, 2.30), (4.00, 2.30), (3.70, 2.30), 
                        (3.70, 2.60), (4.00, 2.60), (4.40, 2.60), (4.80, 2.60), (4.80, 2.80), (4.40, 2.80), (4.00, 2.80), (3.70, 2.80)]
        self._points_idx = 0
        self._landing_region_idx = 0
        self._starting_region_idx = 0
        self._init_points_bool = True

        self._reset_counter = 0


        self.map = OccupancyMap(visualization=self.visualization)

        self._real_theta = 0.0
        self._gamma_theta = 0.1
        self._max_speed = 0.25
        self._min_speed = 0.01

        self._explore_counter = 0
        self._explore_counter_max = 300
        self._explore_yaw_max = np.pi/2
        self._explore_direction = True
        self._explore_speed = 1.0

    # Don't change the method name of 'step_control'
    def step_control(self, sensor_data):

        self.map.updateMap(sensor_data=sensor_data)
              
        # global path planner: state machine
        if self._state == "takeoff":
            real_command = self._takeoff(sensor_data)
            desired_command = real_command
        elif self._state == "search":
            desired_theta, goal_in_polygon = self.map.findPath(sensor_data=sensor_data, goal=self._points[self._points_idx])          
            real_command, desired_command = self._search(sensor_data, desired_theta)
            if goal_in_polygon is not None:
                self._incPointIndex()
                print(f"Goal in polygon: {goal_in_polygon}")
        elif self._state == "explore":
            real_command = self._explore(sensor_data)
            desired_command = real_command 
        elif self._state == "land":
            real_command = self._land(sensor_data)
            desired_command = real_command  
        elif self._state == "reset":
            real_command = self._reset(sensor_data)
            desired_command = real_command
        else:
            raise Exception("Invalid state")
        
        # print(f"next_point: {next_point}")
        
        # draw map to image
        # time_start = time.time()
        if self.visualization:
            self.map.drawMap(sensor_data=sensor_data, real_command=real_command, desired_command=desired_command)
        # print(f"Time for draw map: {time.time()-time_start:.3f} s")
        
        return self._gloabal2local(sensor_data=sensor_data, command=real_command)
        

    def _takeoff(self, sensor_data):
        if sensor_data['range_down'] < self._height_desired-0.01:
            self._height_current += 0.005
            command = [0.0, 0.0, 0.0, self._height_current]
        else:
            command = [0.0, 0.0, 0.0, self._height_desired]
            self._state = "search"
            if self._verb: print("_takeoff: takeoff -> move")
            self._init_points(sensor_data=sensor_data)
        
        return command
    
    def _search(self, sensor_data, desired_theta):

        # check if drone is in landing or starting area
        if ((self._first_part and sensor_data['x_global']>self._landing_region_x[0]) \
            or (not self._first_part and sensor_data['x_global']<self._starting_region_x[1])):
            # check if landing platform is reached
            if sensor_data['range_down'] < 0.42:
                # transition to landing
                self._state = "land"
                if self._verb: print("_search: search -> land")
                return [0.0, 0.0, 0.0, self._height_desired], [0.0, 0.0, 0.0, self._height_desired]

        # check increase point index if next set point is reached
        self._checkPoint(sensor_data=sensor_data, dist=0.05)

        # # check if drone should explore
        # if self.map.shouldExplore:
        #     self._state = "explore"
        #     if self._verb: print("_search: search -> explore")

        # self._explore_counter += 1
        # if self._explore_counter > self._explore_counter_max:
        #     self._state = "explore"
        #     if self._verb: print("_search: search -> explore")
        #     self._explore_counter = 0

        real_command, desired_command = self._theta2command(desired_theta)
        return real_command, desired_command
    
    def _explore(self, sensor_data):
        if self._explore_direction:
            if sensor_data["yaw"] < self._explore_yaw_max:
                return [0.0, 0.0, self._explore_speed, self._height_desired]
            else:
                self._explore_direction = False
                return [0.0, 0.0, -self._explore_speed, self._height_desired]
        else:
            if sensor_data["yaw"] > 0.0:
                return [0.0, 0.0, -self._explore_speed, self._height_desired]
            else:
                self._explore_direction = True
                self._state = "search"
                if self._verb: print("_explore: explore -> search")
                return [0.0, 0.0, 0, self._height_desired]


    
    def _land(self, sensor_data):
        if sensor_data['range_down'] > 0.02:
            self._height_current -= 0.005
            control_command = [0.0, 0.0, 0.0, self._height_current]
        else:
            control_command = [0.0, 0.0, 0.0, 0.0]
            self._state = "reset"
            

            if self._verb:
                print("_land: land -> reset")

        return control_command
    
    def _reset(self, senor_data):
        command = [0.0, 0.0, 0.0, 0.0]

        # return to starting area
        self._reset_counter += 1
        if self._first_part and self._reset_counter > 20:
            self._state = "takeoff"
            self._first_part = False
            self._points_idx = self._starting_region_idx
            self._reset_counter = 0

            if self._verb:
                print("_reset: reset -> takeoff")

        return command
    
    def _theta2command(self, desired_theta):
        # if desired theta is None (no path was found), use last theta
        if desired_theta is None:
            desired_theta = self._real_theta

        # shift desired theta s.t. it is close (|error| <= pi) to real theta
        theta_error = desired_theta - self._real_theta
        if theta_error > np.pi:
            desired_theta -= 2*np.pi
        elif theta_error < -np.pi:
            desired_theta += 2*np.pi
        
        # update real theta with desired theta
        self._real_theta = (1-self._gamma_theta)*self._real_theta + self._gamma_theta*desired_theta

        # calculate real speed: max if real and desired theta are aligned, min if |error| >= pi/2
        real_speed = (1-abs(desired_theta-self._real_theta)/(np.pi/2)) * self._max_speed
        real_speed = np.clip(real_speed, self._min_speed, self._max_speed)
        
        # normalize real theta to (-pi, pi]
        if self._real_theta > np.pi:
            self._real_theta -= 2*np.pi
        elif self._real_theta <= -np.pi:
            self._real_theta += 2*np.pi

        # return real and desired commands
        real_command = [real_speed*np.cos(self._real_theta), real_speed*np.sin(self._real_theta), self._explore_speed, self._height_desired]
        desired_command = [np.cos(desired_theta)*self._max_speed, np.sin(desired_theta)*self._max_speed, self._explore_speed, self._height_desired]     
        return real_command, desired_command
    
    def _init_points(self, sensor_data):
        # remember position of starting area
        if self._init_points_bool:
            x = sensor_data['x_global']
            y = sensor_data['y_global']
            self._points.append((x, y))
            self._starting_region_idx = len(self._points)-1
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
            self._init_points_bool = False
    
    def _gloabal2local(self, sensor_data, command):
        vx = command[0]*np.cos(sensor_data['yaw']) + command[1]*np.sin(sensor_data['yaw'])
        vy = -command[0]*np.sin(sensor_data['yaw']) + command[1]*np.cos(sensor_data['yaw'])
        return [vx, vy, command[2], command[3]]
    
    def _move2point(self, sensor_data, point):
        v_x = point[0] - sensor_data['x_global']
        v_y = point[1] - sensor_data['y_global']
        control_command = [v_x, v_y, 0.0, self._height_desired]
        return control_command
    
    def _dist2point(self, sensor_data, point):
        v_x = point[0] - sensor_data['x_global']
        v_y = point[1] - sensor_data['y_global']
        return np.sqrt(v_x**2 + v_y**2)
    
    def _scaleCommand(self, command, scale):
        norm = np.sqrt(command[0]**2 + command[1]**2)

        if norm == 0:
            return command

        command[0] = command[0]/norm * scale
        command[1] = command[1]/norm * scale
        return command
    
    def _checkPoint(self, sensor_data, dist):

        # increase point index if drone is close enough to the point
        point_is_reached = False
        if self._dist2point(sensor_data, self._points[self._points_idx]) < dist:
            if self._verb:
                print(f"_checkPoint: idx: {self._points_idx}, point: {self._points[self._points_idx]} reached")
        
            self._incPointIndex()
            point_is_reached = True
            print("Go to next point -------------------------------")

        return point_is_reached
    
    def _incPointIndex(self):
        # increase point index
        self._points_idx += 1

        # reset point index if it is out of range
        if self._first_part:           
            if self._points_idx >= self._starting_region_idx:
                self._points_idx = 0
        else:
            if self._points_idx >= len(self._points):
                self._points_idx = self._starting_region_idx
    
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
        

