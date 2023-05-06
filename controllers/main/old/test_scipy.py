import numpy as np
import cv2
from scipy.ndimage import binary_dilation, find_objects, label, generate_binary_structure


class TestSciPy():
    def __init__(self):
        self._threshold = 0.5 # threshold for occupied space
        self._kernel_size = 3 # size of the kernel for morphological operations (must be odd)
        self._contour_approx = 0.02 # approximation of the contours



    def scipy(self, map_array):
        # Threshold the map
        thresholded_map = (map_array > self._threshold).astype(np.uint8)

        # Dilate the obstacles on the map
        structure = np.ones((self._kernel_size, self._kernel_size))
        dilated_map = binary_dilation(thresholded_map, structure=structure).astype(np.uint8)

        # Label the objects in the map
        labeled_map, num_objects = label(dilated_map)

        # Extract the polygons from the labeled objects
        polygons = []
        for i in range(1, num_objects+1):
            # Find the indices of the object in the map
            x_idx = np.where(np.any(labeled_map==i, axis=0))
            y_idx = np.where(np.any(labeled_map==i, axis=1))

            # define polygon as rectangle around object
            polygons.append([(np.min(x_idx), np.min(y_idx)), 
                             (np.max(x_idx), np.min(y_idx)), 
                             (np.max(x_idx), np.max(y_idx)), 
                             (np.min(x_idx), np.max(y_idx))])

        return polygons


    def opencv(self, map_array):
        # Threshold the map
        _, thresholded_map = cv2.threshold(map_array, self._threshold, 1, cv2.THRESH_BINARY)

        # Dilate the obstacles on the map
        dilated_map =   cv2.dilate(
                            src = thresholded_map, 
                            kernel = np.ones((self._kernel_size, self._kernel_size), np.uint8), 
                            iterations = 1
                        )
        
        # Extract the contours of the obstacles from the map using opencv
        contours, _ = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the polygons from the contours using opencv
        polygons = []
        for contour in contours:
            # Approximate the contour with a polygon
            polygon = cv2.approxPolyDP(contour, self._contour_approx * cv2.arcLength(contour, True), True)
            # Convert the polygon to a list of points
            polygon_points = [tuple(point[0]) for point in polygon]
            # Add the polygon to the list of polygons
            polygons.append(polygon_points) 

        return polygons


def test():
    map_array = np.zeros((20,20))
    map_array[10,10] = 1
    map_array[3,3] = 1
    map_array[4,4] = 1

    test = TestSciPy()
    polygons_scipy = test.scipy(map_array)
    polygons_opencv = test.opencv(map_array)

    print(f"polygons_scipy: {polygons_scipy}")
    print(f"polygons_opencv: {polygons_opencv}")

if __name__ == '__main__':
    test()
