import copy
import numpy as np
import skimage
import multiprocessing
from multiprocessing import shared_memory

try:
    from visualization_process import VisualizationProcess
    from navigation import Navigation
    from parameters import Parameters
except:
    from controllers.main.visualization_process import VisualizationProcess
    from controllers.main.navigation import Navigation
    from controllers.main.parameters import Parameters



class Visualization():
    def __init__(self, nav: Navigation, params: Parameters) -> None:
        # navigation object for getting map, polygons, path and start_in_polygon
        self.nav = nav
        self.params = params       

        # initialize image for visualization
        map_copy = np.copy(self.nav.get("map"))
        img_init = 255 * np.ones((map_copy.shape[1], map_copy.shape[0], 3), dtype=np.uint8)

        # create shared memory
        try:
            self.shm = shared_memory.SharedMemory(create=True, size=img_init.nbytes, name='map')
            print("Visualization: Create shared memory")
        except:
            self.shm = shared_memory.SharedMemory(name='map')
            print("Visualization: Shared memory already exists -> open it")
        self.img = np.ndarray(img_init.shape, dtype=img_init.dtype, buffer=self.shm.buf)
        self.img[:] = img_init[:]

        # create parallel process and event
        self.event = multiprocessing.Event()
        x_ticks = np.linspace(0, map_copy.shape[0], 11)
        y_ticks = np.linspace(0, map_copy.shape[1], 7)
        x_labels = np.linspace(self.params.map_size_x[0], self.params.map_size_x[1], 11)
        y_labels = np.linspace(self.params.map_size_y[0], self.params.map_size_y[1], 7)
        labels = (x_ticks, y_ticks, x_labels, y_labels)
        self.process = VisualizationProcess(event=self.event, img_init=img_init, labels=labels)
        self.process.start()

    def __del__(self):
        # free shared memory
        self.shm.close()
        self.shm.unlink()   
    
    def drawMap(self, sensor_data, real_command, desired_command):
        map_copy = np.copy(self.nav.get("map"))
        polygons = copy.deepcopy(self.nav.get("polygons"))
        path = copy.copy(self.nav.get("path"))
        start_in_polygon = self.nav.get("start_in_polygon")

        # create image for visualization
        map_img = np.ones(self.img.shape, dtype=np.uint8) * 255

        # # draw map_array in grayscale
        map_trans = np.clip(np.transpose(map_copy), 0, 1)
        map_img[:, :, 0] = (1-map_trans) * 255
        map_img[:, :, 1] = (1-map_trans) * 255
        map_img[:, :, 2] = (1-map_trans) * 255
        
        # draw polygons in green (or orange if current position is in polygon)
        for idx, poly in enumerate(polygons):
            for i in range(len(poly)):
                if i == len(poly)-1:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[0][0], poly[0][1])        
                else:
                    rr, cc = skimage.draw.line(poly[i][0], poly[i][1], poly[i+1][0], poly[i+1][1])
                rr = np.clip(rr, 0, map_copy.shape[0]-1)
                cc = np.clip(cc, 0, map_copy.shape[1]-1)
                if start_in_polygon == idx: # orange
                    map_img[cc, rr, 0] = 255
                    map_img[cc, rr, 1] = 165
                    map_img[cc, rr, 2] = 0
                else: # green
                    map_img[cc, rr, 0] = 0
                    map_img[cc, rr, 1] = 255
                    map_img[cc, rr, 2] = 0

        # # draw polygon center in orange if current position is inside polygon
        # if start_in_polygon is not None:  
        #     x_mean = np.mean([p[0] for p in polygons[start_in_polygon]], dtype=np.uint32)
        #     y_mean = np.mean([p[1] for p in polygons[start_in_polygon]], dtype=np.uint32)
        #     rr, cc = skimage.draw.ellipse(x_mean, y_mean, r_radius=4, c_radius=4, shape=map_copy.shape)
        #     map_img[cc, rr, 0] = 255
        #     map_img[cc, rr, 1] = 165
        #     map_img[cc, rr, 2] = 0

        # draw path in violett
        for i in range(len(path)-1):
            # print(f"_drawMap: path[i]: {path[i]}, path[i+1]: {path[i+1]}")
            rr, cc = skimage.draw.line(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            rr = np.clip(rr, 0, map_copy.shape[0]-1) 
            cc = np.clip(cc, 0, map_copy.shape[1]-1)
            map_img[cc, rr, 0] = 238
            map_img[cc, rr, 1] = 130
            map_img[cc, rr, 2] = 238       

        # draw drone position and real velocity in blue
        x0 = self.nav._pos2idx(sensor_data['x_global'], "x")
        y0 = self.nav._pos2idx(sensor_data['y_global'], "y")
        x1 = self.nav._pos2idx(sensor_data['x_global'] + real_command[0], "x")
        y1 = self.nav._pos2idx(sensor_data['y_global'] + real_command[1], "y")
        rr_line, cc_line = skimage.draw.line(x0, y0, x1, y1)
        rr_circle, cc_circle, = skimage.draw.ellipse(x0, y0, r_radius=2, c_radius=2, shape=map_copy.shape)
        rr = np.concatenate((rr_line, rr_circle))
        cc = np.concatenate((cc_line, cc_circle))
        map_img[cc, rr, 0] = 0
        map_img[cc, rr, 1] = 0
        map_img[cc, rr, 2] = 255

        # draw desired velocity in red
        x1 = self.nav._pos2idx(sensor_data['x_global'] + desired_command[0], "x")
        y1 = self.nav._pos2idx(sensor_data['y_global'] + desired_command[1], "y")
        rr, cc = skimage.draw.line(x0, y0, x1, y1)
        map_img[cc, rr, 0] = 255
        map_img[cc, rr, 1] = 0
        map_img[cc, rr, 2] = 0

        # save img
        self.img[:] = map_img[:]

        # trigger plotting event
        self.event.set()


