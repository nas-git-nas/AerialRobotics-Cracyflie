import numpy as np
import copy    

from visibility import Visibility
from dijkstra import Dijkstra



class VisibilityGraph():
    def __init__(self) -> None:
        self.vis = Visibility()
        self.dijkstra = Dijkstra()

        self._graph = None # graph array
        self._polygons = None # list of polygons including start [0] and goal [-1] points
        self._poly_idx = None # list of starting indices of each polygone
        self._shortest_path = None # list of points of shortest path  

    def buildGraph(self, polygons, start, goal, boarder_added):
        polygons = copy.deepcopy(polygons)  

        # verify if start is inside a polygon
        if boarder_added:
            ignore_idx = [0, len(polygons)-2, len(polygons)-1] # ignore start, boarder and goal
        else:
            ignore_idx = [0, len(polygons)-1] # ignore start and goal
        start_in_poly_idx = self.vis.insidePolygon(polygons=polygons, point=start, ignore_idx=ignore_idx)
        goal_in_poly_idx = self.vis.insidePolygon(polygons=polygons, point=goal, ignore_idx=ignore_idx)      

        # calculate total number of points and starting index of each polygone
        nb_points = 0
        poly_idx = []
        for poly in polygons:
            poly_idx.append(nb_points)
            nb_points += len(poly)

        # create graph and checked array (which indicates if a connection was already determined)
        graph = np.Inf * np.ones((nb_points,nb_points), dtype=np.float32)
        checked = np.zeros((nb_points,nb_points), dtype=bool)

        # set diagonal to 0
        np.fill_diagonal(graph, 0)
        np.fill_diagonal(checked, True)

        # if start is inside a polygon, add only closest two corners from start to graph
        if start_in_poly_idx:
            graph = self._startInPolyGraph(polygons, start, start_in_poly_idx, poly_idx, graph)
                 
        # loop through all polygons and points
        for i0, poly0 in enumerate(polygons):
            for j0, p0 in enumerate(poly0):
                # loop through all polygons and points
                for i1, poly1 in enumerate(polygons):
                    for j1, p1 in enumerate(poly1):
                        # determine point index for graph array
                        idx0 = poly_idx[i0]+j0
                        idx1 = poly_idx[i1]+j1

                        # continue if graph entry was already determined
                        if checked[idx0, idx1]:
                            continue

                        # indicate that graph entry and inverse were determined
                        checked[idx0, idx1] = True
                        checked[idx1, idx0] = True

                        # continue if start entry was already determined
                        if start_in_poly_idx and (i0 == 0 or i1 == 0):
                            continue

                        # # continue if p1 is in poly0 or p0 is in poly1 (except for border)
                        # if (not boarder_added) or (i0 != len(polygons)-2 and i1 != len(polygons)-2):
                        #     if self.vis.insidePolygon(polygons=[poly0], point=p1, ignore_idx=[ignore_idx]) is not None \
                        #         or self.vis.insidePolygon(polygons=[poly1], point=p0, ignore_idx=[ignore_idx]) is not None:
                        #         continue

                        # calculate distance if p0 is visible to p1
                        if self._isVisible(i0, i1, j0, j1, p0, p1, polygons):                           
                            graph[idx0, idx1] = self._calcDistance(p0, p1)

                        

        self._graph = graph
        self._polygons = polygons
        self._poly_idx = poly_idx

        return start_in_poly_idx, goal_in_poly_idx

    def findShortestPath(self):
        path = self.dijkstra.findShortestPath(self._graph)

        # return empty list if no path was found
        if len(path) == 0:
            self._shortest_path = []
            return []

        # convert path indices to points
        path_points = []
        poly_idx = np.array(self._poly_idx)
        for p in path:
            idx = np.argmax(poly_idx[poly_idx <= p])
            path_points.append(self._polygons[idx][p-poly_idx[idx]])

        self._shortest_path = path_points
        return path_points
    
    def _isVisible(self, i0, i1, j0, j1, p0, p1, polygons):
        # check the easy case first: p0 and p1 are in the same polygone
        if i0 == i1:
            if (abs(j0-j1) == 1) or (abs(j0-j1) == len(polygons[i0])-1):
                return True
            else:
                return False

        # return True if p0 is visible for p1 (and p1 is visible for p0)
        return self.vis.isVisible(p0=p0, p1=p1, polygons=polygons)
    
    def _calcDistance(self, p0, p1):
        return np.linalg.norm([p1[0]-p0[0], p1[1]-p0[1]])
    
    def _startInPolyGraph(self, polygons, start, start_in_poly_idx, poly_idx, graph):
            polygons = copy.deepcopy(polygons)
            start = copy.deepcopy(start)
            start_in_poly_idx = copy.copy(start_in_poly_idx)
            poly_idx = copy.copy(poly_idx)

            # calculate distances from start to all points in polygon
            polygon = np.array(polygons[start_in_poly_idx], dtype=np.uint32)
            xy_min = np.min(polygon, axis=0).astype(np.uint32)
            xy_max = np.max(polygon, axis=0).astype(np.uint32)
            idxs = [np.where((polygon[:,0] == xy_min[0]) & (polygon[:,1] == xy_min[1]))[0][0], # lower left
                    np.where((polygon[:,0] == xy_max[0]) & (polygon[:,1] == xy_min[1]))[0][0], # lower right
                    np.where((polygon[:,0] == xy_max[0]) & (polygon[:,1] == xy_max[1]))[0][0], # upper right
                    np.where((polygon[:,0] == xy_min[0]) & (polygon[:,1] == xy_max[1]))[0][0]] # upper left
            dx = start[0] - xy_min[0]
            dy = start[1] - xy_min[1]
            dx_max = xy_max[0] - xy_min[0]
            dy_max = xy_max[1] - xy_min[1]

            # determine if point 0 (lower left) or point 2 (upper right) is visible
            if dy <= dy_max - (dy_max/dx_max)*dx:
                graph[0, poly_idx[start_in_poly_idx]+idxs[0]] = self._calcDistance(start, polygons[start_in_poly_idx][idxs[0]])
            else:
                graph[0, poly_idx[start_in_poly_idx]+idxs[2]] = self._calcDistance(start, polygons[start_in_poly_idx][idxs[2]])

            # determine if point 1 (lower right) or point 3 (upper left) is visible
            if dy <= (dy_max/dx_max)*dx:
                graph[0, poly_idx[start_in_poly_idx]+idxs[1]] = self._calcDistance(start, polygons[start_in_poly_idx][idxs[1]])
            else:
                graph[0, poly_idx[start_in_poly_idx]+idxs[3]] = self._calcDistance(start, polygons[start_in_poly_idx][idxs[3]])

            return graph

             

# def test_VisibilityGraph():
#     vg = VisibilityGraph()

#     polygons = [[[2,2],[4,2],[4,4],[2,4]], 
#                  [[6,0],[6,1],[7,1],[7,0]],
#                  [[5,1],[6,2],[6,3]]]
#     vg.buildGraph(polygons, start=[0,0], goal=[7,3])
#     vg.findShortestPath()
#     vg.drawGraph()
        
# if __name__ == "__main__":
#     test_VisibilityGraph()


    # def drawGraph(self):
    #     fig, ax = plt.subplots()
    #     self._drawpolygons(ax)
    #     self._drawGraph(ax)
    #     self._drawPath(ax)
    #     plt.show()
    
    # def _drawpolygons(self, ax):
    #     for poly in self._polygons:
    #         x = [p[0] for p in poly] + [poly[0][0]]
    #         y = [p[1] for p in poly] + [poly[0][1]]
    #         ax.plot(x, y, color='black', linewidth=3.0, alpha=0.7)
    
    # def _drawGraph(self, ax):
    #     # calculate total number of points and starting index of each polygone
    #     nb_points = 0
    #     poly_idx = []
    #     for poly in self._polygons:
    #         poly_idx.append(nb_points)
    #         nb_points += len(poly)

    #     # loop through all polygons and points
    #     for i0, poly0 in enumerate(self._polygons):
    #         for j0, p0 in enumerate(poly0):
    #             # loop through all polygons and points
    #             for i1, poly1 in enumerate(self._polygons):
    #                 for j1, p1 in enumerate(poly1):
    #                     # determine point index for graph array
    #                     idx0 = poly_idx[i0]+j0
    #                     idx1 = poly_idx[i1]+j1

    #                     if self._graph[idx0, idx1] < np.Inf:
    #                         ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='blue', linewidth=1.0)

    # def _drawPath(self, ax):
    #     x = [p[0] for p in self._shortest_path]
    #     y = [p[1] for p in self._shortest_path]
    #     ax.plot(x, y, color='red', linewidth=1.75, alpha=1.0)
