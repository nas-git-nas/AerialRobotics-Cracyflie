


class Parameters():

    def __init__(self):
        # general
        self.verb = True
        self.vis = True

        # my controller
        self.mc_gamma_theta = 0.15 # 0.15 # update rate of the heading theta
        self.mc_gamma_speed = 1.0 # update rate of the speed
        self.mc_max_speed = 0.25 # 0.3
        self.mc_min_speed = 0.01
        self.mc_land_speed = 0.05
        self.mc_explore_speed = 1.5 #1.0 # constant yaw speed when searching
        self.mc_height_search = 0.5
        self.mc_height_platform = self.mc_height_search - 0.1
        self.mc_height_delta = 0.07 # height difference for platform detection
        self.mc_min_nb_platform_points = 15
        self.mc_min_dist_platform_reached = 0.03
        self.mc_scan_dist = 0.15
        self.mc_min_control_loop_time = 0.04
        self.mc_land_count_max = int(round(1.5 / self.mc_min_control_loop_time, 0)) 
        self.mc_land_dist = 0.04

        # self.mc_scan_delta_angle = 0.76 #1.57
        # self.mc_first_part_points = [(3.75, 1.40), (3.75, 1.00), (3.75, 0.60), 
        #                             (3.75, 0.25), (4.10, 0.25), (4.45, 0.25), (4.75, 0.25), (4.75, 0.50), (4.40, 0.50), (4.00, 0.50),
        #                             (4.00, 0.80), (4.40, 0.80), (4.75, 0.80), (4.75, 1.10), (4.40, 1.10), (4.00, 1.10),
        #                             (4.00, 1.40), (4.40, 1.40), (4.75, 1.40), (4.75, 1.70), (4.40, 1.70), (4.00, 1.70), (3.75, 1.70),
        #                             (3.75, 2.00), (4.00, 2.00), (4.40, 2.00), (4.75, 2.00), (4.75, 2.25), (4.40, 2.25), (4.00, 2.25), (3.75, 2.25), 
        #                             (3.75, 2.50), (4.00, 2.50), (4.40, 2.50), (4.75, 2.50), (4.75, 2.75), (4.40, 2.75), (4.00, 2.75), (3.70, 2.75)]  
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
        self.map_platform_height = 0.1 # height of the platform in meters

        # nav
        self.nav_range_max = 2 # maximum range of the sensor in meters
        self.nav_alpha = 1.0 # learning rate of new measurements
        self.nav_gamma = 1.0 # discount factor of old measurements
        self.nav_threshold = 0.5 # threshold for occupied space
        self.nav_object_extention = int(round(0.09/self.map_res, 0)) # extention of the objects in meters
        self.nav_kernel_size = int(round(0.13/ self.map_res, 0))
        self.nav_contour_approx = 0.02 # approximation of the contours
        self.nav_polygon_filter_dist = 1.2/self.map_res # filter polygons with a distance smaller than this value
        self.nav_point_reached_dist = 0.06 # in meters, distance to a point to be considered as reached