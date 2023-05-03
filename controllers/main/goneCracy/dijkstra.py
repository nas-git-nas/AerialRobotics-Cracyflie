import numpy as np

class Dijkstra():
    def __init__(self) -> None:
        pass

    def findShortestPath(self, graph):
        dist, prev = self._calcDistance(graph)
        return self._calcPath(dist, prev)

    def _calcDistance(self, graph):
        # initialize arrays
        dist = np.Inf * np.ones((graph.shape[0],), dtype=np.float32) # distance array
        dist[0] = 0
        prev = - np.ones((graph.shape[0],), dtype=np.int32) # preceding point array
        treePoints = np.zeros((graph.shape[0],), dtype=bool) # points included in tree

        for _ in range(treePoints.shape[0]):
            # find point with smallest distance
            idx = np.argmin(np.where(treePoints, np.Inf, dist))
            treePoints[idx] = True

            # update distances
            for i in range(graph.shape[0]):
                # continue if point is already in tree
                if treePoints[i]:
                    continue
                    
                # update distance if it is smaller than previous one
                if dist[idx] + graph[idx,i] < dist[i]:
                    dist[i] = dist[idx] + graph[idx,i]
                    prev[i] = idx

        return dist, prev

    def _calcPath(self, dist, prev):
        # verify if path from start to end point exists
        if dist[-1] == np.Inf:
            return []

        # determine shortest path
        idx = dist.shape[0]-1
        path = [idx]
        while idx != 0:
            idx = prev[idx]
            path.append(idx)

        return np.flip(path)


def test_Dijkstra():
    graph = np.array(  [[0, 4, 0, 0, 0, 0, 0, 8, 0],
                        [4, 0, 8, 0, 0, 0, 0, 11, 0],
                        [0, 8, 0, 7, 0, 4, 0, 0, 2],
                        [0, 0, 7, 0, 9, 14, 0, 0, 0],
                        [0, 0, 0, 9, 0, 10, 0, 0, 0],
                        [0, 0, 4, 14, 10, 0, 2, 0, 0],
                        [0, 0, 0, 0, 0, 2, 0, 1, 6],
                        [8, 11, 0, 0, 0, 0, 1, 0, 7],
                        [0, 0, 2, 0, 0, 0, 6, 7, 0]   ])
    graph = np.where(graph==0, np.Inf, graph)

    d = Dijkstra(graph=graph)
    path = d.findShortestPath()
    print(path)


if __name__ == "__main__":
    test_Dijkstra()


    
    
