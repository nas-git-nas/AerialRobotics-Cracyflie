import numpy as np


class Visibility():
    def __init__(self) -> None:
        pass

    def isVisible(self, p0, p1, polygons):
        # create array that contains all possible line segment that could intersect with p0-q0
        q0 = np.empty((0,2), dtype=np.int32)
        q1 = np.empty((0,2), dtype=np.int32)

        # consider all polygons but only line segments of points that are next to each other
        for poly in polygons:
            q0 = np.concatenate((q0, np.array(poly, dtype=np.int32)), axis=0)
            q1 = np.concatenate((q1, np.roll(np.array(poly, dtype=np.int32), -1, axis=0)), axis=0)

        # remove line segments that contain p0 or p1
        contain_p0 = (q0 == p0).all(axis=1) | (q1 == p0).all(axis=1)
        contain_p1 = (q0 == p1).all(axis=1) | (q1 == p1).all(axis=1)
        q0 = q0[~contain_p0 & ~contain_p1]
        q1 = q1[~contain_p0 & ~contain_p1]       

        # return False if any of the line segments intersect with p0-p1
        p0 = np.tile(p0, (q0.shape[0],1))
        p1 = np.tile(p1, (q0.shape[0],1))
        do_intersect = self._doIntersect(p0, p1, q0, q1)
        return not np.any(do_intersect)
    
    def insidePolygon(self, polygons, point, ignore_idx):
        # verify if point is inside a polygon
        for idx, poly in enumerate(polygons):
            # ignore polygon with index in ignore_idx (start, boarder, goal)
            if idx in ignore_idx:
                continue

            # check if point is inside polygon
            xy_min = np.min(poly, axis=0)
            xy_max = np.max(poly, axis=0)
            if xy_min[0] <= point[0] and point[0] <= xy_max[0] \
                and xy_min[1] <= point[1] and point[1] <= xy_max[1]:
                return idx # polygon index in which the point is inside
            
        return None # point is outside all polygons
    
    def _doIntersect(self, p0, p1, q0, q1):
        # logical array indicating if the line segments intersect
        do_intersect = np.zeros((p0.shape[0]), dtype=bool)
        
        # Find the 4 orientations
        o1 = self._orientation(p0, p1, q0)
        o2 = self._orientation(p0, p1, q1)
        o3 = self._orientation(q0, q1, p0)
        o4 = self._orientation(q0, q1, p1)
    
        # General case
        do_intersect[(o1 != o2) & (o3 != o4)] = True
    
        """ Ignoring special case where line segments """
        # # Special Cases
        # do_intersect[(o1 == 0) & self._onSegment(p0, q0, p1)] = True # p0 , p1 and q0 are collinear and q0 lies on segment p0p1
        # do_intersect[(o2 == 0) & self._onSegment(p0, q1, p1)] = True # p0 , p1 and q1 are collinear and q1 lies on segment p0p1
        # do_intersect[(o3 == 0) & self._onSegment(q0, p0, q1)] = True # q0 , q1 and p0 are collinear and p0 lies on segment q0q1
        # do_intersect[(o4 == 0) & self._onSegment(q0, p1, q1)] = True # q0 , q1 and p1 are collinear and p1 lies on segment q0q1
    
        return do_intersect

    def _onSegment(self, p, q, r):
        # logical array indicating if the point q lies on the line segment p-r
        on_segment = np.zeros((p.shape[0]), dtype=bool)

        # check if q lies on segment p-r
        on_segment[(q[:,0] <= np.maximum(p[:,0], r[:,0])) & (q[:,0] >= np.minimum(p[:,0], r[:,0])) &
                  (q[:,1] <= np.maximum(p[:,1], r[:,1])) & (q[:,1] >= np.minimum(p[:,1], r[:,1]))] = True

        return on_segment
    
    def _orientation(self, p, q, r):
        # array containing the orientation of each triplet (p,q,r)
        orientation = np.zeros((p.shape[0]), dtype=np.int8) # 0: collinear, 1: clockwise, 2: counterclockwise

        # find the orientation of an ordered triplet (p,q,r)
        val = (q[:,1]-p[:,1])*(r[:,0]-q[:,0]) - (q[:,0]-p[:,0])*(r[:,1]-q[:,1])
        orientation[val > 0] = 1 # Clockwise orientation
        orientation[val < 0] = 2 # Counterclockwise orientation

        return orientation  
         

# def test_doIntersect():
#     p0 = np.array([[0,0],[0,0]])
#     p1 = np.array([[4,4],[2,0]])
#     q0 = np.array([[0,4],[0,2]])
#     q1 = np.array([[4,0],[2,2]])

#     v = Visibility()

#     print(f"line segments are crossing: {v._doIntersect(p0, p1, q0, q1)}")

# def test_isVisible():
#     p0 = np.array([4,4])
#     p1 = np.array([5,2])
#     # polygons = [[[2,2],[4,2],[4,4],[2,4]]]
#     # polygons = [[[2,2],[5,2],[5,5],[2,5]], 
#     #              [[5,0],[5,1],[6,1],[6,0]]]
#     polygons = [[[2,2],[5,2],[5,5],[2,5]]]
#     v = Visibility()
#     v.setPointInPolygon(point=(4,4), idx=0)
#     print(f"point is visible: {v.isVisible(p0, p1, polygons)}")

# def test_insidePolygon():
#     point = np.array([5,1])
#     polygons = [[[2,2],[4,2],[4,4],[2,4]], 
#                  [[5,0],[5,1],[6,1],[6,0]],
#                  [[8,8],[8,10],[10,10],[10,8]]]
#     v = Visibility()
#     print(f"point is inside polygon: {v.insidePolygon(point=point, polygons=polygons)}")

# if __name__ == "__main__":
#     # test_doIntersect()
#     test_isVisible()
#     # test_insidePolygon()


# def isVisible(self, p0, p1, polygons):
#         # create array that contains all possible line segment that could intersect with p0-q0
#         q0 = np.empty((0,2), dtype=np.int32)
#         q1 = np.empty((0,2), dtype=np.int32)

#         # determine if p0 or p1 is inside a polygon
#         if self._in_polygon_point is not None \
#             and ((p0[0]==self._in_polygon_point[0] and p0[1]==self._in_polygon_point[1]) 
#                  or (p1[0]==self._in_polygon_point[0] and p1[1]==self._in_polygon_point[1])):
#             # only consider polygon that contains p0 or p1 but all possible line segments
#             for i in range(len(polygons[self._in_polygon_idx])):
#                 q0 = np.concatenate((q0, np.array(polygons[self._in_polygon_idx], dtype=np.int32)), axis=0)
#                 q1 = np.concatenate((q1, np.roll(np.array(polygons[self._in_polygon_idx], dtype=np.int32), -i, axis=0)), axis=0)

#             # print(f"q0: {q0}")
#             # print(f"q1: {q1}")
#         else:
#             # consider all polygons but only line segments of points that are next to each other
#             for poly in polygons:
#                 q0 = np.concatenate((q0, np.array(poly, dtype=np.int32)), axis=0)
#                 q1 = np.concatenate((q1, np.roll(np.array(poly, dtype=np.int32), -1, axis=0)), axis=0)

#         # remove line segments that contain p0 or p1
#         contain_p0 = (q0 == p0).all(axis=1) | (q1 == p0).all(axis=1)
#         contain_p1 = (q0 == p1).all(axis=1) | (q1 == p1).all(axis=1)
#         q0 = q0[~contain_p0 & ~contain_p1]
#         q1 = q1[~contain_p0 & ~contain_p1]       

#         # return False if any of the line segments intersect with p0-p1
#         p0 = np.tile(p0, (q0.shape[0],1))
#         p1 = np.tile(p1, (q0.shape[0],1))
#         do_intersect = self._doIntersect(p0, p1, q0, q1)
#         return not np.any(do_intersect)


    # def insidePolygon(self, polygons, point):
    #     # create array that contains all possible line segment that could intersect with p0-q0
    #     q0 = np.empty((0,2), dtype=np.int32)
    #     q1 = np.empty((0,2), dtype=np.int32)
    #     nb_points = 0
    #     poly_idx = []
    #     for poly in polygons:
    #         q0 = np.concatenate((q0, np.array(poly, dtype=np.int32)), axis=0)
    #         q1 = np.concatenate((q1, np.roll(np.array(poly, dtype=np.int32), -1, axis=0)), axis=0)
    #         poly_idx.append(nb_points)
    #         nb_points += len(poly)

    #     # create line segment from point to the right (negative y direction)
    #     p0 = np.tile(point, (q0.shape[0],1))
    #     p1 = np.tile((point[0],0), (q0.shape[0],1))

    #     # determine if point is on polygon outline
    #     onSegment = self._onSegment(q0, p0, q1)
    #     # print(f"onSegment: {onSegment}, disc. {poly_idx < np.argmax(onSegment)}, idx: {np.argmax(onSegment)}, poly_idx: {poly_idx}")
    #     if np.any(onSegment):
    #         # return index of polygon on which the point is on
    #         idx = np.argmax(onSegment) # point index of first True value
            
    #         # convert point index to polygon index
    #         for j in range(len(poly_idx)-1):
    #             if poly_idx[j] <= idx and idx < poly_idx[j+1]:
    #                 return j 
    #         return len(poly_idx)-1

    #     # determine if point is inside polygon
    #     do_intersect = self._doIntersect(p0, p1, q0, q1)
    #     # print(f"do_intersect: {do_intersect}")
    #     for idx in range(len(poly_idx)):
    #         # determine range of polygon indices
    #         if idx == len(poly_idx)-1:
    #             idx_range = (poly_idx[idx], len(do_intersect))
    #         else:
    #             idx_range = (poly_idx[idx], poly_idx[idx+1])
            
    #         # count number of intersections with polygon
    #         nb_intersections = np.sum(do_intersect[idx_range[0]:idx_range[1]])
    #         # print(f"idx: {idx}, nb_intersections: {nb_intersections}, range: {idx_range}")
    #         if nb_intersections % 2 == 1:
    #             return idx # polygon index
            
    #     return False # point is outside all polygons