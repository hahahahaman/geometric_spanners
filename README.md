# Geometric Spanner Algorithms

## Plan
1. Implement an algorithm that determines the stretch factor of any
given geometric graph.

2. Implement the algorithm that constructs the spanner mentioned
   above. plot the graph to see if it works

3. Run experiments on point sets in convex position to determine the
stretch factor ratio of the spanner mentioned above.

4. Analyze the results of the experiments and make observations about
the pairs of points that attain the stretch factor.

5. Use the observations from 4. to improve the upper and lower bounds on
the stretch factor.

6. Use the knowledge obtained to design new algorithms for constructing
plane spanners for points that are not in convex position. 

## Definitions

geometric graph: G(S, E), where S is a non-empty finite set of points
in space, E is a set of edges that connect a pair of points, (p,q),
such that the weight of the edge is equal to the distance between the
points, |pq|

minimum path length: the minimum path length of two vertices, p,q is
denoted by |pq|_G, which is the shortest path from p to q

spanner: a graph in which all vertices are connected; there is a path
connecting any two vertices

t-spanner: a graph such that for any two points, p,q, |pq|_G <= t *
|pq|

stretch factor: the minimum value of t for a t-spanner of the graph

plane spanner: spanner whose edges do not cross when drawn on a plane

convex position: a set of points is in convex position if all the
points are on the convex hull formed from the points

chain: a path, sequence of consecutive points connected by edges

diametrical pair: the largest distance between any pair of points in
a graph is the diameter. a diametrical pair is any pair with distance
equal to the diameter

Theorem: given a fixed compact convex shape with k vertices, C, let
Xn be the random sample of n points chosen uniformly and
independently from inside C. let Zn denote the number of vertices of
the convex hull of Xn, then E[Zn] = O(k*logn).
Reference: Har-Peled, On the Expected Complexity of Random Convex Hulls

## License

[MIT](LICENSE.txt)

[3rd party license](LICENSE_3RD_PARTY.txt)
