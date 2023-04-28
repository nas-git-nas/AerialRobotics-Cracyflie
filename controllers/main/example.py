# Examples of basic methods for simulation competition
import numpy as np
import matplotlib.pyplot as plt

# Global variables
on_ground = True
_height_desired = 0.5

# Obstacle avoidance with range sensors
def obstacle_avoidance(sensor_data):
    global on_ground, _height_desired

    # Take off
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, 0.0, _height_desired]
        return control_command
    else:
        on_ground = False

    # Obstacle avoidance with distance sensors
    if sensor_data['range_front'] < 0.2:
        if sensor_data['range_left'] > sensor_data['range_right']:
            control_command = [0.0, 0.2, 0.0, _height_desired]
        else:
            control_command = [0.0, -0.2, 0.0, _height_desired]
    else:
        control_command = [0.2, 0.0, 0.0, _height_desired]

    return control_command

# Coverage path planning
setpoints = [[-0.0, 0.0], [-0.0, -2.0], [-0.5, -2.0], [-0.5, 0.0]]
index_current_setpoint = 0
def path_planning(sensor_data):
    global on_ground, _height_desired, index_current_setpoint, setpoints

    # Take off
    if on_ground and sensor_data['range_down'] < 0.49:
        control_command = [0.0, 0.0, 0.0, _height_desired]
        return control_command
    else:
        on_ground = False

    # Hover at the final setpoint
    if index_current_setpoint == len(setpoints):
        control_command = [0.0, 0.0, 0.0, _height_desired]
        return control_command

    # Get the goal position and drone position
    x_goal, y_goal = setpoints[index_current_setpoint]
    x_drone, y_drone = sensor_data['x_global'], sensor_data['y_global']
    distance_drone_to_goal = np.linalg.norm([x_goal - x_drone, y_goal- y_drone])

    # When the drone reaches the goal setpoint, e.g., distance < 0.1m
    if distance_drone_to_goal < 0.1:
        # Select the next setpoint as the goal position
        index_current_setpoint += 1
        # Hover at the final setpoint
        if index_current_setpoint == len(setpoints):
            control_command = [0.0, 0.0, 0.0, _height_desired]
            return control_command

    # Calculate the control command based on current goal setpoint
    x_goal, y_goal = setpoints[index_current_setpoint]
    x_drone, y_drone = sensor_data['x_global'], sensor_data['y_global']
    v_x, v_y = x_goal - x_drone, y_goal - y_drone
    control_command = [v_x, v_y, 0.0, _height_desired]
    return control_command
    
# Occupancy map based on distance sensor
min_x, max_x = 0, 5.0 # meter
min_y, max_y = 0, 5.0 # meter
range_max = 2.0 # meter, maximum range of distance sensor
res_pos = 0.2 # meter
conf = 0.2 # certainty given by each measurement
t = 0 # only for plotting

map = np.zeros((int((max_x-min_x)/res_pos), int((max_y-min_y)/res_pos))) # 0 = unknown, 1 = free, -1 = occupied

def occupancy_map(sensor_data):
    global map, t
    pos_x = sensor_data['x_global']
    pos_y = sensor_data['y_global']
    yaw = sensor_data['yaw']
    
    for j in range(4): # 4 sensors
        yaw_sensor = yaw + j*np.pi/2 #yaw positive is counter clockwise
        if j == 0:
            measurement = sensor_data['range_front']
        elif j == 1:
            measurement = sensor_data['range_left']
        elif j == 2:
            measurement = sensor_data['range_back']
        elif j == 3:
            measurement = sensor_data['range_right']
        
        for i in range(int(range_max/res_pos)): # range is 2 meters
            dist = i*res_pos
            idx_x = int(np.round((pos_x - min_x + dist*np.cos(yaw_sensor))/res_pos,0))
            idx_y = int(np.round((pos_y - min_y + dist*np.sin(yaw_sensor))/res_pos,0))

            # make sure the point is within the map
            if idx_x < 0 or idx_x >= map.shape[0] or idx_y < 0 or idx_y >= map.shape[1] or dist > range_max:
                break

            # update the map
            if dist < measurement:
                map[idx_x, idx_y] += conf
            else:
                map[idx_x, idx_y] -= conf
                break
    
    map = np.clip(map, -1, 1) # certainty can never be more than 100%

    # only plot every Nth time step (comment out if not needed)
    if t % 50 == 0:
        plt.imshow(np.flip(map,1), vmin=-1, vmax=1, cmap='gray', origin='lower') # flip the map to match the coordinate system
        plt.savefig("map.png")
    t +=1

    return map