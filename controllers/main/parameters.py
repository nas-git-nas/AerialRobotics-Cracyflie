import numpy as np


class Parameters():

    def __init__(self):
        """
        GENERAL
        """
        self.verb = True
        self.vis = True
        self.simulation = False
        self.control_loop_period = 0.03 # in seconds, period of the main loop       

        """
        PATH
        - all points for takeoff, search and return
        """
        self.path_init_pos = (1.0, 1.0) # position of takeoff
        self.path_first_point = (3.9,1.5) # (3.0,0.5) first point to fly to
        # self.path_search_points_lower = [(3.5,0.5), (4.0,0.5), (4.0,1.0), (3.5,1.0)]
        # self.path_search_points_upper = self.path_search_points_lower
        self.path_search_points_lower = [ (3.9,1.1), (3.9,0.7), # points from the lower (y<1.5m) search zone
                                        (3.9,0.4), (4.25,0.4), (4.6,0.4), (4.6,0.65), (4.4,0.65), (4.15,0.65),
                                        (4.15,0.7), (4.4,0.7), (4.6,0.7), (4.6,0.95), (4.4,0.95), (4.15,0.95),
                                        (4.15,1.2), (4.4,1.2), (4.6,1.2), (4.6,1.4), (4.4,1.4), (3.9,1.5) ]
        self.path_search_points_upper = [ (3.9,1.9), (3.9,2.3), # points from the upper (y>=1.5m) search zone
                                        (3.9,2.6), (4.25,2.6), (4.6,2.6), (4.6,2.35), (4.4,2.35), (4.15,2.35),
                                        (4.15,2.1), (4.4,2.1), (4.6,2.1), (4.6,1.85), (4.4,1.85), (4.15,1.85),
                                        (4.15,1.6), (4.4,1.6), (4.6,1.6), (4.6,1.4), (4.4,1.4), (3.9,1.5) ]
        
        """
        SEARCH
        - state 'search' in MyController
        """
        self.sea_alpha_yaw = 0.3 #0.15 # update rate of the heading theta
        self.sea_alpha_speed = 1.0 # update rate of the speed
        self.sea_alpha_height = 0.8 # update rate of the height
        self.sea_speed_max = 0.15 #0.25 
        self.sea_speed_min = 0.02 #0.01
        self.sea_height_ground = 0.3
        self.sea_height_platform = self.sea_height_ground - 0.1
        self.sea_height_delta = 0.08 # height difference for platform detection
        self.sea_point_reached_dist = 0.1 # distance to point when it is considered reached

        """
        EXPLORE
        - state 'explore' in MyController
        """
        self.explore_speed_max = 0.05 # in meters/seconds, maximum speed to keep position while exploring
        self.explore_speed_min = 0.02 # in meters/seconds, minimum speed to keep position while exploring
        self.explore_yaw_speed = 0.35 # constant yaw speed when searching
        self.explore_yaw_error = 0.05
        self.explore_counter_max_init = 100
        self.explore_counter_delta = 0


        """
        DETECT_PAD
        """
        

        """
        LAND
        - state land in MyController
        """
        self.land_speed = 0.05
        self.land_point_reached_dist = 0.07 # in meters, continue to move in the same direction for this distance

        """
        RESET
        """
        self.reset_counter_max = 15

        """
        MAP
        - map related constants in Navigation
        """
        self.map_res = 0.05 # resolution of the map in meters 
        self.map_size_x = (0, 5) # map x limits in meters
        self.map_size_y = (0, 3) # map y limits in meters
        self.map_landing_region_x = (3.5, 5) # x range of landing region
        self.map_landing_region_y = (0, 3) # y range of landing region
        self.map_starting_region_x = (0, 1.5) # x range of starting region
        self.map_starting_region_y = (0, 3) # y range of starting region
        self.map_boarder_size = int(round(0.05/self.map_res, 0)) # size of the boarder in pixels

        """
        NAVIGATION
        - navigation related constants in Navigation
        """
        self.nav_range_max = 1.5 # maximum range of the sensor in meters
        self.nav_alpha = 0.55 # update rate of new measurements
        self.nav_gamma = 1.0 # discount factor of old measurements
        self.nav_threshold = 0.5 # threshold for occupied space
        self.nav_object_extention = int(round(0.08/self.map_res, 0)) # extention of the objects in meters
        self.nav_kernel_size = int(round(0.13/ self.map_res, 0))
        self.nav_polygon_filter_dist = 1.2/self.map_res # filter polygons with a distance smaller than this value
        self.nav_point_reached_dist = 0.1 # in meters, distance to a point to be considered as reached
        self.nav_yaw_correction = 0.71 # in radians, yaw correction to move away from obstacle if drone is inside polygon

        """
        CRACYFLIE
        - crazyflie related constans in CrazyflieConnection
        """
        self.crazy_alpha = 0.7 # updating rate of crazyflie's horizontal range sensors


        """
        SIMULATION
        - change some values to make work the simulation
        """
        if self.simulation:
            self.sea_speed_max = 0.2
            self.explore_counter_max_init = 125
            self.nav_object_extention = int(round(0.08/self.map_res, 0)) # extention of the objects in meters
            self.explore_speed_max = 0.0
            self.explore_speed_min = 0.0