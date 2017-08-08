

#TODO: Modular approach, verify somethings
#Vipin meet weekend
#Read Paper
#D3, Kaggle


from igraph import *



import numpy as np
from scipy import sparse
import numpy.linalg as linalg


g = Graph.Read_Adjacency('assort_graph.txt', sep=',', comment_char='#', attribute="weight")
print g.es["weight"]

#modularity
# calculate dendrogram
dendrogram = g.community_edge_betweenness()
# convert it into a flat clustering
clusters = dendrogram.as_clustering()
# get the membership vector
membership = clusters.membership


#clmembership = g.community_fastgreedy().as_clustering()

print g.modularity(membership, weights=g.es["weight"])

#clustering coefficient
print g.transitivity_avglocal_undirected(mode='nan', weights=g.es["weight"])




# average path length (Doubtful)
def get_averagePathLength(mygraph, weight_attribute=False):
    """Calculate the average path length of a graph"""
    if(weight_attribute != False):
        weighted = True
    else:
        weighted = False

    current_lengths = []

    if weighted is True and weight_attribute not in mygraph.es.attributes():
        print("attribute", weight_attribute, "does not exists")
   
    for node in mygraph.vs():
        if weighted is True:
            current_lengths.extend([len(node_paths)-1 for node_paths in mygraph.get_shortest_paths(node, weights=mygraph.es[weight_attribute]) if len(node_paths)>1])
        else:
            current_lengths.extend([len(node_paths)-1 for node_paths in mygraph.get_shortest_paths(node) if len(node_paths)>1])

    av_path_length = mean(current_lengths)
    return av_path_length

#observed = get_averagePathLength(g,g.es["weight"])
expected = g.average_path_length()
#print observed, expected
print expected
#g.average_path_length(directed=True, unconn=True)

#g.pathlenth()

#efficiency (Doubtful)
print 1.0/expected



#Assortativity (Doubtful check)
print g.assortativity_degree(directed=False)



#laplacian
Lap_mat = g.laplacian(weights=g.es["weight"], normalized=False)
#values = eigen(Lap_mat)
D, V = linalg.eig(Lap_mat)
print(D)
A=sorted(D)
#Fiedler value
print(A[1])




#Normalized Laplacian
Lap_mat2 = g.laplacian(weights=g.es["weight"], normalized=True)
D, V = linalg.eig(Lap_mat2)
print(D)
A=sorted(D)
#Normalized Fiedler value
print(A[1])


#Average Degree
print mean(Graph.degree(g)) / 2

#Connection Density
print g.density(loops = False)


#betweenness centrality / hub centrality
print g.betweenness()
print mean(g.betweenness())









