from __future__ import generators
import numpy as np
import matplotlib.pyplot as plt
import math,random

random.seed()

n = 10

def random_points(n):
    pnts = []
    for i in range(n):
        pnts += [(random.random(),random.random())]
    return pnts

# points = np.random.rand(n, 2) # 30 random points in 2D, chosen inside a square
points = random_points(n)
# points = [[0,0], [0.5,0], [0.5,0.5], [0.0,0.5]]

# convex hull (monotone chain by x-coordinate) and diameter of a set of points
# David Eppstein, UC Irvine, 7 Mar 2002


def orientation(p,q,r):
    '''Return positive if p-q-r are clockwise, neg if ccw, zero if colinear.'''
    return (q[1]-p[1])*(r[0]-p[0]) - (q[0]-p[0])*(r[1]-p[1])

def hulls(Points):
    '''monotone chain to find upper and lower convex hulls of a set of 2d points.'''
    U = []
    L = []
    # Points = Points[np.lexsort((Points[:,1],Points[:,0]))]
    Points.sort()
    for p in Points:
        while len(U) > 1 and orientation(U[-2],U[-1],p) <= 0: U.pop()
        while len(L) > 1 and orientation(L[-2],L[-1],p) >= 0: L.pop()
        U.append(p)
        L.append(p)
    return U,L

def rotatingCalipers(U, L):
    '''Given a list of 2d points, finds all ways of sandwiching the points
between two parallel lines that touch one point each, and yields the sequence
of pairs of points touched by each pair of lines.'''
    i = 0
    j = len(L) - 1
    while i < len(U) - 1 or j > 0:
        yield U[i],L[j] # generated values

        # if all the way through one side of hull, advance the other side
        if i == len(U) - 1: j -= 1
        elif j == 0: i += 1

        # still points left on both lists, compare slopes of next hull edges
        # being careful to avoid divide-by-zero in slope calculation
        elif (U[i+1][1]-U[i][1])*(L[j][0]-L[j-1][0]) > \
                (L[j][1]-L[j-1][1])*(U[i+1][0]-U[i][0]):
            i += 1
        else: j -= 1

def diameter(U,L):
    '''Given a list of 2d points, returns the pair that's farthest apart.'''
    diam,pair = max([((p[0]-q[0])**2 + (p[1]-q[1])**2, (p,q))
                     for p,q in rotatingCalipers(U,L)])
    return pair

U,L = hulls(points)

dpair = diameter(U,L)
# U = U[1:]
# L.pop()
# L+=U[1:]

print(points)
print()
print(U)

for p in points:
    plt.plot(p[0], p[1], marker='o', color='b')

def plot_edges(s, color='k', linestyle='-'):
    for i in range(len(s)-1):
        plt.plot([s[i][0], s[i+1][0]], [s[i][1], s[i+1][1]], color=color, linestyle=linestyle)

plot_edges(U, color='g')
plot_edges(U[1:], color='k', linestyle='--')
plot_edges(L, color='m')
plot_edges(dpair, color='r', linestyle=':')

plt.show()
