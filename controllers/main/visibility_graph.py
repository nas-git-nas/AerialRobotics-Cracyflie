import numpy as np
import copy
import matplotlib.pyplot as plt

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

    def buildGraph(self, polygons, start, goal):
        # add start and goal point to polygons
        polygons = copy.deepcopy(polygons)
        polygons = [[start]] + polygons + [[goal]]

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

                        # calculate distance if p0 is visible to p1
                        if self._isVisible(i0, i1, j0, j1, p0, p1, polygons):                           
                            graph[idx0, idx1] = self._calcDistance(p0, p1)

                        # indicate that graph entry and inverse were determined
                        checked[idx0, idx1] = True
                        checked[idx1, idx0] = True

        self._graph = graph
        self._polygons = polygons
        self._poly_idx = poly_idx

    def findShortestPath(self):
        path = self.dijkstra.findShortestPath(self._graph)

        # convert path indices to points
        path_points = []
        poly_idx = np.array(self._poly_idx)
        for p in path:
            idx = np.argmax(poly_idx[poly_idx <= p])
            path_points.append(self._polygons[idx][p-poly_idx[idx]])

        self._shortest_path = path_points
        return path_points

    def drawGraph(self):
        fig, ax = plt.subplots()
        self._drawpolygons(ax)
        self._drawGraph(ax)
        self._drawPath(ax)
        plt.show()
    
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
    
    def _drawpolygons(self, ax):
        for poly in self._polygons:
            x = [p[0] for p in poly] + [poly[0][0]]
            y = [p[1] for p in poly] + [poly[0][1]]
            ax.plot(x, y, color='black', linewidth=3.0, alpha=0.7)
    
    def _drawGraph(self, ax):
        # calculate total number of points and starting index of each polygone
        nb_points = 0
        poly_idx = []
        for poly in self._polygons:
            poly_idx.append(nb_points)
            nb_points += len(poly)

        # loop through all polygons and points
        for i0, poly0 in enumerate(self._polygons):
            for j0, p0 in enumerate(poly0):
                # loop through all polygons and points
                for i1, poly1 in enumerate(self._polygons):
                    for j1, p1 in enumerate(poly1):
                        # determine point index for graph array
                        idx0 = poly_idx[i0]+j0
                        idx1 = poly_idx[i1]+j1

                        if self._graph[idx0, idx1] < np.Inf:
                            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color='blue', linewidth=1.0)

    def _drawPath(self, ax):
        x = [p[0] for p in self._shortest_path]
        y = [p[1] for p in self._shortest_path]
        ax.plot(x, y, color='red', linewidth=1.75, alpha=1.0)
             

def test_VisibilityGraph():
    vg = VisibilityGraph()

    polygons = [[[2,2],[4,2],[4,4],[2,4]], 
                 [[6,0],[6,1],[7,1],[7,0]],
                 [[5,1],[6,2],[6,3]]]
    vg.buildGraph(polygons, start=[0,0], goal=[7,3])
    vg.findShortestPath()
    vg.drawGraph()
        
if __name__ == "__main__":
    test_VisibilityGraph()
