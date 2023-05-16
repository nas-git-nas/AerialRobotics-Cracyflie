# You can change anything in this file except the file name of 'my_control.py',
# the class name of 'MyController', and the method name of 'step_control'.

# Available sensor data includes data['t'], data['x_global'], data['y_global'],
# data['roll'], data['pitch'], data['yaw'], data['v_forward'], data['v_left'],
# data['range_front'], data['range_left'], data['range_back'],
# data['range_right'], data['range_down'], data['yaw_rate'].

import numpy as np
import time

try:
    from navigation import Navigation
    from parameters import Parameters
    from visualization import Visualization
except:
    from controllers.main.navigation import Navigation
    from controllers.main.parameters import Parameters
    from controllers.main.visualization import Visualization


# Don't change the class name of 'MyController'
class MyController():
    def __init__(self, params: Parameters=None):
        # Initialize the parameters
        if params is None:
            self.params = Parameters()
        else:
            self.params = params

        # Initialize the navigation and visualization
        self.nav = Navigation(params=self.params)
        if self.params.vis:
            self.vis = Visualization(nav=self.nav, params=self.params)

        """
        State machine:
            - takeoff
            - search (search for landing or starting platform)
            - explore (turn the drone to explore map)
            - land
            - reset (either transition from first to second part or shutdown the drone)
        """
        self._state = "takeoff"

        """
        Two parts:
            True: drone goes from starting area to landing platform
            False: drone goes from landing area to starting platform
        """
        self._first_part = True

        # self._search_points = [(3.9,1.5)] # list of points to visit
        self._search_points = [self.params.path_first_point] # list of points to visit
        self._search_points_idx = 0 # index of the current point
        self._init_position = None # initial position of the drone when taking off

        self._applied_yaw = 0.0 # applied yaw of the drone
        self._applied_speed = 0.0 # applied speed of the drone
        self._applied_height = 0.0 # applied height of the drone

        self._land_pos =(0.0, 0.0) # position where landing platform was detected
        self._land_vel = (0.0, 0.0) # velocity of the drone when landing platform was detected

        self._explore_yaw = 0.0 # yaw goal of the drone when exploring
        self._explore_pos = (0.0, 0.0) # (x,y) position while exploring (keep it steady)
        self._explore_counter = self.params.explore_counter_max_init # counter for exploring
        self._explore_counter_max = self.params.explore_counter_max_init # maximum counter for exploring

        self._reset_counter = 0 # counter for approaching ground during reset (only in first part)

    def setLanding(self):
        """
        Force landing with keyboard interrupt
        """
        # transition to landing
        if self._applied_height > 0.02:
            if self.params.verb and (self._state != "land"): 
                print("setLanding: keyboard -> land")
            self._state = "land"
        else:
            if self.params.verb and (self._state != "reset"): 
                print("setLanding: keyboard -> reset")
            self._state = "reset"

    def step_control(self, sensor_data):
        """
        One step of the main control loop. Contains the state machine,
        draws the map and converts the command from the global to the 
        local reference frame.
            :param sensor_data: measurement data from crazyflie, dict
            :return local_command: command in local reference frame (vx, vy, yaw, height), list
        """
        # measure time of entire control loop
        step_start_time = time.time()

        # global path planner: state machine
        desired_command = [0.0, 0.0, 0.0, 0.0] # dummy command for visualization
        if self._state == "takeoff":
            real_command = self._takeoff(sensor_data)
        elif self._state == "search":                 
            real_command, desired_command = self._search(sensor_data)
        elif self._state == "explore":
            real_command, desired_command = self._explore(sensor_data)      
        elif self._state == "land":
            real_command = self._land(sensor_data)
        elif self._state == "reset":
            real_command = self._reset()
        else:
            raise Exception("Invalid state")
        
        # visualize map
        if self.params.vis:
            self.vis.drawMap(sensor_data=sensor_data, real_command=real_command, desired_command=desired_command)

        # convert command from global to local reference frame
        local_command = self._gloabal2local(sensor_data=sensor_data, command=real_command)

        # sleep to ensure no rush (only in simulation, with crazyflie this is done in the main)
        if self.params.simulation:
            step_time = time.time() - step_start_time
            if step_time < self.params.control_loop_period:
                time.sleep(self.params.control_loop_period - step_time)

        return local_command
        

    def _takeoff(self, sensor_data):
        """
        First state of state machine used to takeoff. Transitions always to the state 'search'.
            :param sensor_data: measurement data from crazyflie, dict
            :return command: command in global reference frame (vx, vy, yaw, height), list
        """
        if sensor_data['range_down'] < self.params.sea_height_ground-0.03:
            self._applied_height += np.maximum(0.005 * (self.params.sea_height_ground-sensor_data['range_down'])/self.params.sea_height_ground, 0.002)
            command = [0.0, 0.0, 0.0, self._applied_height]
        else:
            command = [0.0, 0.0, 0.0, self._applied_height]
            self._state = "search"
            if self.params.verb: print("_takeoff: takeoff -> search")

            if self._init_position is None and self._first_part:
                self._init_position = (sensor_data["x_global"], sensor_data["y_global"])
        
        return command
    
    def _search(self, sensor_data):
        """
        State of the state machine used to find landing platform. In the first part the
        drone moves towards the landing region and searches there for the platform. In the
        second part the drone moves towards the starting region and searches there for the
        platform. Transitions to state 'land' if platform is found. Otherwise, it transitions
        to state 'explore' if explore counter reached its max. and the drone is not inside any
        polygon.
            :param sensor_data: measurement data from crazyflie, dict
            :return real_command: command in global reference frame applied to the drone (vx, vy, vyaw, height), list
            :return desired_command: command in global reference frame that would be desired (before the smoothing), list
        """
        # Update the map
        self.nav.updateMap(sensor_data=sensor_data)

        # check if drone is over platform     
        if self._platformTransition(sensor_data=sensor_data):
            # transition to landing
            self._state = "land"
            if self.params.verb: print("_search: search -> land")

            # calculate landing velocity and position where platform was detected
            self._land_vel = (np.cos(self._applied_yaw)*self.params.land_speed, np.sin(self._applied_yaw)*self.params.land_speed)
            self._land_pos = (sensor_data["x_global"], sensor_data["y_global"])

            # stop moving
            command = [self._land_vel[0], self._land_vel[1], 0.0, self._applied_height]
            return command, command

        # find shortest path to next point
        desired_yaw, goal_in_polygon, start_in_polygon = self.nav.findPath(sensor_data=sensor_data, goal=self._search_points[self._search_points_idx]) 
        
        # check if drone should explore
        self._explore_counter += 1
        if (self._explore_counter > self._explore_counter_max) and (not start_in_polygon):
            # reset counter
            self._explore_counter = 0
            self._explore_counter_max += self.params.explore_counter_delta

            # remember position to keep it constant during exploring
            self._explore_pos = (sensor_data["x_global"], sensor_data["y_global"])

            # set yaw to which drone turns while exploring
            if self._explore_yaw == 0.0: # turn for the first time more
                self._explore_yaw = sensor_data["yaw"] + np.pi/2 
            else:
                self._explore_yaw = sensor_data["yaw"] + np.pi/3         

            # transition to explore
            self._state = "explore"
            if self.params.verb: print("_search: search -> explore")
            command = [0.0, 0.0, 0.0, self._applied_height]
            return command, command

        # check increase point index if next set point is reached
        self._checkPoint(sensor_data=sensor_data, dist=self.params.sea_point_reached_dist, goal_in_polygon=goal_in_polygon)  

        # convert yaw to command
        real_command, desired_command = self._yaw2command(
            desired_yaw=desired_yaw, 
            yaw_speed=0.0,
            goal_in_polygon=goal_in_polygon, 
            speed_max=self.params.sea_speed_max, 
            speed_min=self.params.sea_speed_min,
        )

        # return desired and real command
        return real_command, desired_command
    
    def _explore(self, sensor_data):
        """
        State of the state machine used to explore the map. Turns the drone around the z-axis 
         by a certain amount and then, transitions back to state 'search'.
            :param sensor_data: measurement data from crazyflie, dict
            :return real_command: command in global reference frame applied to the drone (vx, vy, vyaw, height), list
            :return desired_command: command in global reference frame that would be desired (before the smoothing), list
        """
        # Update the map
        self.nav.updateMap(sensor_data=sensor_data)
        
        # normalize explore yaw with respect to real yaw
        if self._explore_yaw - sensor_data["yaw"] > np.pi:
            self._explore_yaw -= 2*np.pi
        elif self._explore_yaw - sensor_data["yaw"] <= -np.pi:
            self._explore_yaw += 2*np.pi
        
        # check if drone turned enough
        if abs(self._explore_yaw - sensor_data["yaw"]) < self.params.explore_yaw_error:
            # transition to search
            self._state = "search"
            if self.params.verb: print("_explore: explore -> search")

            command = [0.0, 0.0, 0.0, self._applied_height]
            return command, command

        # determine desired yaw to keep position steady while turning
        desired_yaw = np.arctan2(self._explore_pos[1]-sensor_data["y_global"], self._explore_pos[0]-sensor_data["x_global"])

        # convert yaw to command
        real_command, desired_command = self._yaw2command(
            desired_yaw=desired_yaw, 
            yaw_speed=self.params.explore_yaw_speed,
            goal_in_polygon=None, 
            speed_max=self.params.explore_speed_max, 
            speed_min=self.params.explore_speed_min,
        )

        # return desired and real command
        return real_command, desired_command
    
    def _land(self, sensor_data):
        """
        State of the state machine used to land the droen. First, drone continuous to move in the same direction
        as when transitioning from state 'search' to 'land' for a certain distance to make sure that
        it does not land on the edge of the platform. Second, the drone lands and transitions to the 
        state 'reset'.
            :param sensor_data: measurement data from crazyflie, dict
            :return command: command in global reference frame (vx, vy, vyaw, height), list
        """     
        # continue moving in same direction for a fixed distance
        if self._dist2point(sensor_data=sensor_data, point=self._land_pos) < self.params.land_point_reached_dist:
            # update applied height
            self._updateAppliedHeight(desired_height=self.params.sea_height_platform)

            return [self._land_vel[0], self._land_vel[1], 0.0, self._applied_height]
        
        # land
        if sensor_data['range_down'] > 0.03:
            self._applied_height -= np.maximum(0.01 * (sensor_data['range_down'])/self.params.sea_height_ground, 0.004)
        else:
            self._state = "reset"
            if self.params.verb: print("_land: land -> reset")
        
        return [0.0, 0.0, 0.0, self._applied_height]
    
    def _reset(self):
        """
        State of the state machine used to reset the droen. If the drone is in the first part, it will transition to
        the state 'takeoff' and search the starting platform. If the drone is in the second part, it will shutdown.
            :return command: command in global reference frame (vx, vy, vyaw, height), list
        """
        # return to starting area
        if self._first_part:
            # smoothly approach ground for certain amount of time
            self._reset_counter += 1
            if self._reset_counter > self.params.reset_counter_max:
                self._first_part = False
                self._search_points = [self._init_position]
                self._search_points_idx = 0

                self._state = "takeoff"
                if self.params.verb: print("_reset: reset -> takeoff")
        
            # update applied height
            # self._updateAppliedHeight(desired_height=0.0)
            # return [0.0, 0.0, 0.0, self._applied_height]
            return (0.0, 0.0, 0.0, 0.0)

        # second part: shut down
        return (0.0, 0.0, 0.0, 0.0)
    
    def _yaw2command(self, desired_yaw, yaw_speed, goal_in_polygon, speed_max, speed_min):
        """
        Converts the desired yaw into a command. Makes a smooth approximation between the last
        applied yaw and the desired one to prevent command jittering.
            :param desired_yaw: desired yaw in radians, float
            :param yaw_speed: yaw rate in degrees per seconds, float
            :param goal_in_polygon: indicates in which polygon the goal is (None if not inside any polygon), int
            :param speed_max: maximum applied horizontal speed, float
            :param speed_min: minimum applied horizontal speed, float
            :return real_command: command in global reference frame applied to the drone (vx, vy, vyaw, height), list
            :return desired_command: command in global reference frame that would be desired (before the smoothing), list
        """
        # if desired theta is None (no path was found), use last theta
        if desired_yaw is None:
            desired_yaw = self._applied_yaw
            if self.params.verb:
                print(f"_yaw2command: No path was found, using last yaw={np.round(np.rad2deg(desired_yaw),1)}°")

        # if goal is inside polygon, use last theta
        if goal_in_polygon is not None:
            desired_yaw = self._applied_yaw
            if self.params.verb:
                print(f"_yaw2command: Goal inside polygon, using last yaw={np.round(np.rad2deg(desired_yaw),1)}°")
        
        # update real theta with desired theta
        self._updateAppliedYaw(desired_yaw=desired_yaw)

        # calculate real speed: max if real and desired theta are aligned, min if |error| >= pi/2
        desired_speed = (1-abs(desired_yaw-self._applied_yaw)/(np.pi/2)) * speed_max
        desired_speed = np.clip(desired_speed, speed_min, speed_max)
        self._updateAppliedSpeed(desired_speed=desired_speed)       

        # update applied height
        self._updateAppliedHeight(desired_height=self.params.sea_height_ground)

        # return real and desired commands
        real_command = [self._applied_speed*np.cos(self._applied_yaw), self._applied_speed*np.sin(self._applied_yaw), yaw_speed, self._applied_height]
        desired_command = [np.cos(desired_yaw)*self.params.sea_speed_max, np.sin(desired_yaw)*self.params.sea_speed_max, yaw_speed, self._applied_height] 
        return real_command, desired_command
    
    def _updateAppliedYaw(self, desired_yaw):
        """
        Make a smooth approximation (exponential moving average) of the applied yaw by using the desired yaw.
            :param desired_yaw: desired yaw in radians
        """
        # normalize applied yaw to (-pi, pi]
        if self._applied_yaw > np.pi:
            self._applied_yaw -= 2*np.pi
        elif self._applied_yaw <= -np.pi:
            self._applied_yaw += 2*np.pi

        # shift desired yaw s.t. it is close to real applied yaw (|error| <= pi)
        theta_error = desired_yaw - self._applied_yaw
        if theta_error > np.pi:
            desired_yaw -= 2*np.pi
        elif theta_error <= -np.pi:
            desired_yaw += 2*np.pi

        # update applied yaw
        self._applied_yaw = (1-self.params.sea_alpha_yaw) * self._applied_yaw + self.params.sea_alpha_yaw * desired_yaw
    
    def _updateAppliedSpeed(self, desired_speed):
        """
        Make a smooth approximation (exponential moving average) of the applied speed by using the desired speed.
            :param desired_speed: desired horizontal speed in meters per seconds
        """
        self._applied_speed = (1-self.params.sea_alpha_speed) * self._applied_speed + self.params.sea_alpha_speed * desired_speed
    
    def _updateAppliedHeight(self, desired_height):
        """
        Make a smooth approximation (exponential moving average) of the applied height by using the desired height.
            :param desired_height: desired height in meters
        """
        self._applied_height = (1-self.params.sea_alpha_height) * self._applied_height + self.params.sea_alpha_height * desired_height
    
    def _platformTransition(self, sensor_data):
        """
        Detect a when the drone transitions from the searching area to a platform.
            :param sensor_data: measurement data from crazyflie, dict
            :return isTrue: wheather or not the drone encountered a transition, bool
        """
        # check if drone is in landing or starting region
        if ((self._first_part and sensor_data['x_global'] < self.params.map_landing_region_x[0]) \
            or (not self._first_part and sensor_data['x_global'] > self.params.map_starting_region_x[1])):
            return False
        
        # if self._applied_height-sensor_data['range_down'] > self.params.sea_height_delta:
        if self.params.sea_height_ground-sensor_data['range_down'] > self.params.sea_height_delta:
            self._applied_height = sensor_data['range_down']
            if self.params.verb:
                print("_platformTransition: transition detected")
            return True
        
        return False
    
    def _gloabal2local(self, sensor_data, command):
        """
        Covnert the command from the global to the local reference frame. Note that positive yaw is 
        defined by the crazyflie in the opposite direction than by this code and therefore, it is
        inverted. Also, the yaw rate is converted from radians per seconds to degrees per second.
        These transformations are not necessary in simulation.
            :param sensor_data: measurement data from crazyflie, dict
            :return command: command in global reference frame (vx, vy, vyaw, height), list
            :return local_command: command in local reference frame (vx, vy, vyaw, height), list
        """
        vx = command[0]*np.cos(sensor_data['yaw']) + command[1]*np.sin(sensor_data['yaw'])
        vy = -command[0]*np.sin(sensor_data['yaw']) + command[1]*np.cos(sensor_data['yaw'])

        if self.params.simulation:
            vyaw = command[2]
        else:
            vyaw = - (command[2]/np.pi) * 180
        return [vx, vy, vyaw, command[3]]
    
    def _dist2point(self, sensor_data, point):
        """
        Calculate the distance between the current drone position and a point.
            :param sensor_data: measurement data from crazyflie, dict
            :param point: point on map (x,y), tuble
        """
        v_x = point[0] - sensor_data['x_global']
        v_y = point[1] - sensor_data['y_global']
        return np.sqrt(v_x**2 + v_y**2)
    
    def _checkPoint(self, sensor_data, dist, goal_in_polygon):
        """
        Verify if the drone reached its goal and set the next setpoint as goal in this case.
        If the goal is inside a polygon, continue with the next setpoint as well.
            :param sensor_data: measurement data from crazyflie, dict
            :param goal_in_polygon: indicates in which polygon the goal is (None if not inside any polygon), int
            :param dist: critical distance, goal is considered to be reached if drone is closer, float
            :return goal_reached: True if drone is closer than dist w.r.t. goal, otherwise return False, bool
        """
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
        """
        Set next setpoint as goal by increasing the point index. If the end of the search points is reached, 
        then add points from the current search area. Hence, this will be search points from the landing area 
        during the first part and search points from the starting area during the second part. Before adding
        the search points during the first part, the algorithm checks if there are more obstacles in the upper
        part of the landing region (y>=1.5) or in the lower part (y<1.5). Then, the drone explores first the
        region with less obstacles.
        """
        # continue searching if not all points are yet explored
        self._search_points_idx += 1
        if self._search_points_idx < len(self._search_points):
            return

        # reset search points if the counter is out of range
        self._search_points_idx = 0

        # second part: return to starting platform
        if not self._first_part: 
            self._search_points = [self._init_position]
            return
        else: # first part
        
            # first part: search for landing platform
            # count the number of polygons in the lower and upper part of the landing region
            upper_counter, lower_counter = self._countObstaclesInLowerUpperParts()

            # search first the part with fewer polygons
            if upper_counter > lower_counter:
                self._search_points = self.params.path_search_points_lower + self.params.path_search_points_upper
            else:
                self._search_points = self.params.path_search_points_upper + self.params.path_search_points_lower

    def _countObstaclesInLowerUpperParts(self):
        """
        Count the number of obstacles in the upper part of the landing region (y>=1.5) 
         and in the lower part (y<1.5) to decide where to search first for the platform.
            :return upper_counter: number of obstacles with y >= 1.5 and x > 3.5, int
            :return lower_counter: number of obstacles with y < 1.5 and x > 3.5, int
        """
        # get all (unfiltered polygons)
        polygons = self.nav.get('unfiltered_polygons')

        # count polygons in upper and lower landing region
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

        return upper_counter, lower_counter



















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
    #     real_command, desired_command = self._theta2command(desired_theta, max_speed=self.params.land_speed)

    #     return real_command, desired_command

    # def _init_scan_points(self, sensor_data):
    #     # rotational matrix in direction of movement
    #     cos_theta = np.cos(self._applied_yaw)
    #     sin_theta = np.sin(self._applied_yaw)
    #     rot_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    #     # scanning displacement
    #     dx = self.params.mc_scan_dist
    #     dy = self.params.mc_scan_dist
    #     delta_dist = np.array([ (dx,0), (dx,dy), (0,dy),
    #                             (0,-dy), (-dx,-dy), (-dx,0),
    #                             (2*dx,0), (2*dx,-2*dy), (0,-2*dy),
    #                             (0,2*dy), (-2*dx,2*dy), (-2*dx,0),
    #                             (3*dx,0), (3*dx,3*dy), (0,3*dy),
    #                             (0,-3*dy), (-3*dx,-3*dy), (-3*dx,0),
    #                             (4*dx,0), (4*dx,-4*dy), (0,-4*dy),
    #                             (0,4*dy), (-4*dx,4*dy), (-4*dx,0),])
    #     delta_dist = np.transpose(np.matmul(rot_matrix, np.transpose(delta_dist)))

    #     # scanning center point
    #     x = sensor_data['x_global']
    #     y = sensor_data['y_global']
    #     pos = np.tile(np.array([x, y]), (delta_dist.shape[0], 1))

    #     self._platform_points = []
    #     self._scan_points_idx = 0
    #     self._scan_points = pos + delta_dist
    #     self._scan_points[:,0] = np.clip(self._scan_points[:,0], 
    #                                      self.params.map_size_x[0] + 1.2*self.params.map_boarder_size*self.params.map_res, 
    #                                      self.params.map_size_x[1] - 1.2*self.params.map_boarder_size*self.params.map_res)
    #     self._scan_points[:,1] = np.clip(self._scan_points[:,1], 
    #                                      self.params.map_size_y[0] + 1.2*self.params.map_boarder_size*self.params.map_res, 
    #                                      self.params.map_size_y[1] - 1.2*self.params.map_boarder_size*self.params.map_res)

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
        

