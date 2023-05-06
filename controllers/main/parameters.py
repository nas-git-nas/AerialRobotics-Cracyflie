


class Parameters():

    def __init__(self):
        # general
        self.verb = True
        self.vis = True

        # my controller
        self.mc_gamma_theta = 0.15 # 0.15 # update rate of the heading theta
        self.mc_gamma_speed = 1.0 # update rate of the speed
        self.mc_max_speed = 0.3 # 0.3
        self.mc_min_speed = 0.01
        self.mc_land_speed = 0.1
        self.mc_explore_speed = 1.5 #1.0 # constant yaw speed when searching
        self.mc_height_search = 0.5
        self.mc_height_platform = 0.1
        self.mc_height_scan = self.mc_height_search - self.mc_height_platform
        self.mc_height_delta = 0.03 # height difference for platform detection
        self.mc_min_nb_platform_points = 10
        self.mc_min_dist_platform_reached = 0.03
        self.mc_scan_delta_radius = 1.1*self.mc_min_dist_platform_reached
        self.mc_scan_delta_angle = 0.76 #1.57
        self.mc_first_part_points = [(3.75, 1.40), (3.75, 1.00), (3.75, 0.60), 
                                    (3.75, 0.25), (4.10, 0.25), (4.45, 0.25), (4.75, 0.25), (4.75, 0.50), (4.40, 0.50), (4.00, 0.50),
                                    (4.00, 0.80), (4.40, 0.80), (4.75, 0.80), (4.75, 1.10), (4.40, 1.10), (4.00, 1.10),
                                    (4.00, 1.40), (4.40, 1.40), (4.75, 1.40), (4.75, 1.70), (4.40, 1.70), (4.00, 1.70), (3.75, 1.70),
                                    (3.75, 2.00), (4.00, 2.00), (4.40, 2.00), (4.75, 2.00), (4.75, 2.25), (4.40, 2.25), (4.00, 2.25), (3.75, 2.25), 
                                    (3.75, 2.50), (4.00, 2.50), (4.40, 2.50), (4.75, 2.50), (4.75, 2.75), (4.40, 2.75), (4.00, 2.75), (3.70, 2.75)]       
        
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
        object_extention = 0.15 # extention of the objects in meters
        self.nav_kernel_size = int(round(2*object_extention / self.map_res, 0))
        self.nav_contour_approx = 0.02 # approximation of the contours
        self.nav_polygon_filter_dist = 0.8/self.map_res # filter polygons with a distance smaller than this value
        self.nav_point_reached_dist = 0.04 # in meters, distance to a point to be considered as reached