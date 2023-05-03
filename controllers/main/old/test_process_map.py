import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyvisgraph as vg


class ProcessMap():
    def __init__(self):
        pass

    def __call__(self, obstacle_map):

        # Converting the numpy array into image
        obstacle_map  = obstacle_map.astype(np.float32)
        

        # Set the threshold value for the map
        threshold_value = 0.5

        # Threshold the map
        thresholded_map = cv2.threshold(obstacle_map, threshold_value, 1, cv2.THRESH_BINARY)[1]

        # Dilate the obstacles on the map
        dilated_map = cv2.dilate(thresholded_map, np.ones((1, 1), np.uint8), iterations=1)

        print(f"dilated_map: \n{dilated_map}")

        # Extract the contours of the obstacles from the map using opencv
        contours, hierarchy = cv2.findContours(dilated_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract the polygons from the contours using opencv
        polygons = []
        for contour in contours:
            # Approximate the contour with a polygon
            polygon = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
            # Convert the polygon to a list of points
            polygon_points = [tuple(point[0]) for point in polygon]
            # Add the polygon to the list of polygons
            polygons.append(polygon_points)

        # Print the polygons
        print(polygons)

        # convert polygons to pyvisgraph points
        vg_polygons = []
        for poly in polygons:
            vg_polygons.append([vg.Point(p[0], p[1]) for p in poly])

        # Create visibility graph
        graph = vg.VisGraph()
        graph.build(vg_polygons)

        # Get shortest path between two points
        start = vg.Point(0, 0)
        end = vg.Point(6, 6)
        path = graph.shortest_path(start, end)

        # Print path
        print(path)

        # Plot polygons
        fig, ax = plt.subplots()

        for poly in vg_polygons:
            xs = []
            ys = []
            for point in poly:
                xs.append(point.x)
                ys.append(point.y)
            xs.append(poly[0].x)
            ys.append(poly[0].y)
                
            ax.plot(xs, ys, color='black')

        # Plot path
        xs = []
        ys = []
        for point in path:
            xs.append(point.x)
            ys.append(point.y)

        xs = np.round(xs, 0).astype(np.int32)
        ys = np.round(ys, 0).astype(np.int32)
        ax.plot(xs, ys, color='red')


        # Show plot
        plt.show()


    
def test_process_map():
    # Define the map containing obstacles as a numpy array
    # obstacle_map = np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 1, 1, 0, 0],
    #     [0, 1, 1, 0, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 1, 0]
    # ])
    obstacle_map = np.array([
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0]
    ])

    # Process map
    process_map = ProcessMap()
    process_map(obstacle_map)



if __name__ == "__main__":
    test_process_map()