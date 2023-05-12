


class Parameters():

    def __init__(self):
        # general
        self.verb = True
        self.vis = True

        # my controller
        self.mc_gamma_theta = 0.15 # 0.15 # update rate of the heading theta
        self.mc_gamma_speed = 1.0 # update rate of the speed
        self.mc_gamma_height = 0.4 # update rate of the height
        self.mc_max_speed = 0.25 
        self.mc_min_speed = 0.01
        self.mc_land_speed = 0.05
        self.mc_yaw_speed = 1.0 # constant yaw speed when searching
        self.mc_height_search = 0.5
        self.mc_height_platform = self.mc_height_search - 0.1
        self.mc_height_delta = 0.07 # height difference for platform detection
        self.mc_min_control_loop_time = 0.04
        self.mc_land_dist = 0.04
        self.mc_explore_min_vel = 0.02
        self.mc_explore_counter_max_init = 150
        self.mc_explore_counter_delta = 75
        self.mc_explore_yaw_error = 0.01
        self.mc_search_points_lower = [ (3.9,1.1), (3.9,0.7),
                                        (3.9,0.4), (4.25,0.4), (4.6,0.4), (4.6,0.65), (4.4,0.65), (4.15,0.65),
                                        (4.15,0.7), (4.4,0.7), (4.6,0.7), (4.6,0.95), (4.4,0.95), (4.15,0.95),
                                        (4.15,1.2), (4.4,1.2), (4.6,1.2), (4.6,1.4), (4.4,1.4), (3.9,1.5) ]
        self.mc_search_points_upper = [ (3.9,1.9), (3.9,2.3),
                                        (3.9,2.6), (4.25,2.6), (4.6,2.6), (4.6,2.35), (4.4,2.35), (4.15,2.35),
                                        (4.15,2.1), (4.4,2.1), (4.6,2.1), (4.6,1.85), (4.4,1.85), (4.15,1.85),
                                        (4.15,1.6), (4.4,1.6), (4.6,1.6), (4.6,1.4), (4.4,1.4), (3.9,1.5) ]
        
        # map
        self.map_res = 0.01 # resolution of the map in meters 
        self.map_size_x = (0, 5) # map x limits in meters
        self.map_size_y = (0, 3) # map y limits in meters
        self.map_landing_region_x = (3.5, 5) # x range of landing region
        self.map_landing_region_y = (0, 3) # y range of landing region
        self.map_starting_region_x = (0, 1.5) # x range of starting region
        self.map_starting_region_y = (0, 3) # y range of starting region
        self.map_boarder_size = int(round(0.1/self.map_res, 0)) # size of the boarder in pixels

        # nav
        self.nav_range_max = 2 # maximum range of the sensor in meters
        self.nav_alpha = 1.0 # learning rate of new measurements
        self.nav_gamma = 1.0 # discount factor of old measurements
        self.nav_threshold = 0.5 # threshold for occupied space
        self.nav_object_extention = int(round(0.09/self.map_res, 0)) # extention of the objects in meters
        self.nav_kernel_size = int(round(0.13/ self.map_res, 0))
        self.nav_polygon_filter_dist = 1.2/self.map_res # filter polygons with a distance smaller than this value
        self.nav_point_reached_dist = 0.06 # in meters, distance to a point to be considered as reached