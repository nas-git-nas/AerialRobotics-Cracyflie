

# import pyvisgraph as vg


# polys = [[vg.Point(0.0,1.0), vg.Point(3.0,1.0), vg.Point(1.5,4.0)],
#          [vg.Point(4.0,4.0), vg.Point(7.0,4.0), vg.Point(5.5,8.0)]]
# g = vg.VisGraph()
# g.build(polys)
# shortest = g.shortest_path(vg.Point(1.5,0.0), vg.Point(4.0, 6.0))

# print(shortest)


import pyvisgraph as vg
import matplotlib.pyplot as plt

# Define vertices of polygons
polygon1 = [(2, 2), (2, 3), (3, 3), (3, 2)]
polygon2 = [(5, 5), (5, 15), (15, 15), (15, 5)]

polygon1 = [vg.Point(p[0], p[1]) for p in polygon1]
polygon2 = [vg.Point(p[0], p[1]) for p in polygon2]

# Create visibility graph
graph = vg.VisGraph()
graph.build([polygon1, polygon2])

# Get shortest path between two points
start = vg.Point(0, 0)
end = vg.Point(20, 20)
path = graph.shortest_path(start, end)

# Print path
print(path)

# Plot polygons
fig, ax = plt.subplots()

for poly in [polygon1, polygon2]:
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
ax.plot(xs, ys, color='red')


# Show plot
plt.show()