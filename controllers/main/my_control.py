# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np
import time

from occupancy_map import OccupancyMap

# Don't change the class name of 'MyController'
class MyController():
    def __init__(self):
        self._verb = True

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


        landing_region_points = [(3.70, 0.20), (4.80, 0.20), (4.80, 0.50), (3.70, 0.50),
                                    (3.70, 0.80), (4.80, 0.80), (4.80, 1.10), (3.70, 1.10),
                                    (3.70, 1.40), (4.80, 1.40), (4.80, 1.70), (3.70, 1.70),
                                    (3.70, 2.00), (4.80, 2.00), (4.80, 2.30), (3.70, 2.30)]
        starting_region_points = [(1.30, 0.20), (0.20, 0.20), (0.20, 0.50), (1.30, 0.50),
                                    (1.30, 0.80), (0.20, 0.80), (0.20, 1.10), (1.30, 1.10),
                                    (1.30, 1.40), (0.20, 1.40), (0.20, 1.70), (1.30, 1.70),
                                    (1.30, 2.00), (0.20, 2.00), (0.20, 2.30), (1.30, 2.30)]
        self._points = landing_region_points + starting_region_points
        self._points_idx = 0
        self._landing_region_idx = 0
        self._starting_region_idx = len(landing_region_points)

        # # self._obstacle_delay = 20 # number of steps to delay after obstacle avoidance
        # # self._obstacle_counter = 0
        # # self._last_command = [0.0, 0.0, 0.0, 0.0]
        # # self._last_sensors = [False, False, False, False]
        # self._avoid_critical = [False, False, False, False]
        # self._avoid_direction = 1
        # self._state_before_avoid = "takeoff"

        # self._reset_counter = 0
        # self._avoid_counter = 0

        self._map = OccupancyMap()

    # Don't change the method name of 'step_control'
    def step_control(self, sensor_data):

        print("my_control: step_control")
        # time.sleep(1)

        next_point, goal_in_obstacle = self._map.step(sensor_data, goal=self._points[self._points_idx])

        if goal_in_obstacle:
            self._incPointIndex()

        print(f"next_point: {next_point}")
        
        # global path planner: state machine
        if self._state == "takeoff":
            command = self._takeoff(sensor_data)
        # elif self._state == "move":
        #     command = self._move(sensor_data, next_point)
        elif self._state == "search":
            command = self._search(sensor_data, next_point)
        elif self._state == "land":
            command = self._land(sensor_data)
        # elif self._state == "avoid":
            # local path planner: obstacle avoidance
            # command = self._avoid(sensor_data)          
        elif self._state == "reset":
            command = self._reset(sensor_data)
        else:
            raise Exception("Invalid state")
        
        # print(f"command: {command}")

        
        
        return command
        

    def _takeoff(self, sensor_data):
        # remember position of starting area
        if self._first_part:
            self._points = self._points[:self._starting_region_idx] \
                            + [(sensor_data['x_global'], sensor_data['y_global'])] \
                            + self._points[self._starting_region_idx:]

        if sensor_data['range_down'] < self._height_desired-0.01:
            self._height_current += 0.005
            command = [0.0, 0.0, 0.0, self._height_current]
        else:
            command = [0.0, 0.0, 0.0, self._height_desired]
            self._state = "search"
            if self._verb:
                print("_takeoff: takeoff -> move")
        
        return command
    
    def _search(self, sensor_data, next_point):

        # check if drone is in landing or starting area
        if ((self._first_part and sensor_data['x_global']>self._landing_region_x[0]) \
            or (not self._first_part and sensor_data['x_global']<self._starting_region_x[1])):
            # check if landing platform is reached
            if sensor_data['range_down'] < 0.42:
                # transition to landing
                self._state = "land"
                if self._verb: print("_search: search -> land")
                return [0.0, 0.0, 0.0, self._height_desired]

        # move towards next set point    
        command = self._move2point(sensor_data, next_point)
        command = self._scaleCommand(command=command, scale=0.2)

        # check increase point index if next set point is reached
        self._checkPoint(sensor_data=sensor_data, dist=0.05)

        return command
    
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
        # self._reset_counter += 1
        
        # if self._first_part and self._reset_counter > 100:
        #     self._state = "takeoff"
        #     self._first_part = False
        #     self._points_idx = self._starting_region_idx
        #     self._reset_counter = 0

        #     if self._verb:
        #         print("_reset: reset -> takeoff")

        return command
    
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
        

