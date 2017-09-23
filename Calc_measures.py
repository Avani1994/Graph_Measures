import library_graph_measuresv1 as lib

from igraph import *

import numpy as np
from scipy import sparse
import numpy.linalg as linalg
import os
import math
from random import *





#g = Graph.Read_Adjacency('/Users/avanisharma/Masters/2nd_Sem/Research Project/HCP/HCP_Untransformed_distmat/100307dist.txt', sep=',', comment_char='#', attribute="weight")

g = Graph.Read_Adjacency('assort_graph_eg1.txt', sep=',', comment_char='#', attribute="weight")
weightss = g.es["weight"]
vertices = len(g.vs)
vertexlist = range(vertices)
lists = g.get_adjlist()
print "Modularity:"
print lib.modularity(g,weightss)
print "Clustering Coefficient:"
print lib.clusteringCoefficient(g,weightss)
print "Average Path Length:"
print lib.avgPathLength(g)
print "Assortativity:"
print lib.assortativity(g)
print "Fiedler Value:"
print lib.fiedlerValue(g,weightss)
print "Normalized Fiedler Value:"
print lib.normalizedFiedler(g,weightss)
print "Average Degree:"
print lib.avgDegree(g)
print "Connection Density:"
print lib.conDensity(g)
print "Betweenness Centrality:"
# just divide the output by two : or dont if considering revrse paths to (1-2,2-1)
print lib.betweennessCentrality(g)
# This last procedure calculates wrong verified by internet example
#print lib.brandes(vertexlist,lists)