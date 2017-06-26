import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import math

n = 10
points = np.random.rand(n, 2) # 30 random points in 2D, chosen inside a square
hull = ConvexHull(points)

# print(2 * math.log(n, 2)) # approximately 2*logn vertices are part of the hull
# print([points[v][0] for v in hull.vertices])

print(hull.simplices)

plt.plot(points[:,0], points[:,1], 'o')

for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()
