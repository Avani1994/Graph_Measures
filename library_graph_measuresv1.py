

#TODO: Modular approach, verify somethings
#Read Paper
#D3, Kaggle


from igraph import *



import numpy as np
from scipy import sparse
import numpy.linalg as linalg
from collections import deque



#modularity verified (without weights graph)
#https://en.wikipedia.org/wiki/Modularity_(networks) - verify through this too
#https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/modularity.pdf - to understand
#https://www.researchgate.net/post/Can_anyone_provide_a_short_example_of_how_the_modularity_is_being_calculated_in_networks -: verify through this
# answer = 0.8
def modularity(g,weight):
    # calculate dendrogram
    dendrogram = g.community_edge_betweenness()
    # convert it into a flat clustering
    clusters = dendrogram.as_clustering()
    # get the membership vector
    membership = clusters.membership
    print membership
    
    #clmembership = g.community_fastgreedy().as_clustering()

    return g.modularity(membership, weights = weight)
    



#clustering coefficient (Verified without weights)
#https://en.wikipedia.org/wiki/Clustering_coefficient
#http://qasimpasta.info/data/uploads/sina-2015/calculating-clustering-coefficient.pdf
def clusteringCoefficient(g,weight):
    #****** related to weights ********#
    #print g.transitivity_local_undirected(mode='zero',weights=weight)
    #print g.transitivity_local_undirected(mode='zero')
    #****** related to weights ********#
    return g.transitivity_avglocal_undirected(mode='zero')




#avgpathlength verified  (without weights)
# For fully connected graph without weights should be 1
def avgPathLength(g): 
    expected = g.average_path_length()
    #print observed, expected
    
    #g.average_path_length(directed=True, unconn=True)

    #g.pathlenth()

    #efficiency (Doubtful)
    #print 1.0/expected
    return expected
    


#Assortativity (Doubtful check) without weights verfied
#https://www.slideshare.net/jaquechocolate/complex-networks-assortativity
#https://en.wikipedia.org/wiki/Assortativity
#https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=0ahUKEwiE5JKQnsbVAhUKwmMKHb6ED3AQjRwIBw&url=http%3A%2F%2Fwww.scielo.org.mx%2Fscielo.php%3Fpid%3DS1870-90442016000100005%26script%3Dsci_arttext&psig=AFQjCNFo0DmgydZuhOAlrcfm9jgtSH_EDw&ust=1502233520512139
#def assortativity(g):
    #return g.assortativity_degree()
    
#http://igraph.wikidot.com/python-recipes
#http://snipplr.com/view/9914/ (included in igraph python)
def assortativity(g, degrees=None):
    if degrees is None: degrees = g.degree()
    degrees_sq = [deg**2 for deg in degrees]
 
    m = float(g.ecount())
    num1, num2, den1 = 0, 0, 0
    
    for source, target in g.get_edgelist():
        
        num1 += degrees[source] * degrees[target]
        num2 += degrees[source] + degrees[target]
        den1 += degrees_sq[source] + degrees_sq[target]
    
    num1 /= m
    den1 /= 2*m
    num2 = (num2 / (2*m)) ** 2

    if(den1 - num2 == 0):
        return "NaN"
    else:
        return (num1 - num2) / (den1 - num2)

#self implementation
def assortativity(g,degrees = None):
    if degrees is None: degrees = g.degree()
    degree_source_list = []
    degree_target_list = []
    for source, target in g.get_edgelist():
        degree_source_list = degree_source_list + [degrees[source] - 1]
        degree_target_list = degree_target_list + [degrees[target] - 1]
    degree_source_np = np.asarray(degree_source_list)
    degree_target_np = np.asarray(degree_target_list)
    return corr(degree_source_np,degree_target_np)


def corr(data1, data2):
    "data1 & data2 should be numpy arrays."
    mean1 = data1.mean() 
    mean2 = data2.mean()
    std1 = data1.std()
    std2 = data2.std()

#   corr = ((data1-mean1)*(data2-mean2)).mean()/(std1*std2)
    corr = ((data1*data2).mean()-mean1*mean2)/(std1*std2)
    return corr


#laplacian
#Fiedler value
#https://en.wikipedia.org/wiki/Algebraic_connectivity (Example to verify wikipedia)
def fiedlerValue(g,weight):
    Lap_mat = g.laplacian(weights=weight, normalized=False)
    #values = eigen(Lap_mat)
    #print Lap_mat
    D, V = linalg.eig(Lap_mat)
    #print(D)
    #print V
    A=sorted(D)

    return A[1]
    








#Normalized Laplacian 
#Normalized Fiedler Value
#http://www.sciencedirect.com/science/article/pii/S0024379514001748 (Definition)
def normalizedFiedler(g,weight):
    Lap_mat2 = g.laplacian(weights=weight, normalized=True)
    #print Lap_mat2
    D, V = linalg.eig(Lap_mat2)
    #print(D) 
    #print V
    A=sorted(D)
    #Normalized Fiedler value
    return A[1]
    




#Average Degree verfied without weights
#For fully connected should be number of nodes - 1
#Doubt y it is giving total edges = 2*edges ???
def avgDegree(g):
    #print len(g.es)
    #print len(g.vs)
    return (float(len(g.es))) / float(len(g.vs))



#Connection Density verified without weights
#For fully connected should be 1
def conDensity(g):
    vertices = len(g.vs)
    edges = len(g.es)
    print float(edges) / float((vertices * (vertices - 1)))
    return g.density(loops = False)
    




#betweenness centrality / hub centrality seems to be correct not sure
#https://www.sci.unich.it/~francesc/teaching/network/betweeness.html (Not sure but yes this graph should give non 0 value)
#http://med.bioinf.mpi-inf.mpg.de/netanalyzer/help/2.7/index.html#nodeBetween - Example to check
#https://www.youtube.com/watch?v=ptqt2zr9ZRE (betweenness centrality of graph example of two graphs) for verification, 
#library method does the job, division by 2 ?? if both edges - no , if one edge - yes 
def betweennessCentrality(g):
    #print g.betweenness() 
    arr = g.betweenness(nobigint=False)
    betweenness = [elem/2.0 for elem in arr]
    print betweenness
    return mean(betweenness)


#code from net : https://stackoverflow.com/questions/23660696/betweennes-centrality-python
def brandes(V, A):
    "Compute betweenness centrality in an unweighted graph."
    # Brandes algorithm
    # see http://www.cs.ucc.ie/~rb4/resources/Brandes.pdf
    C = dict((v,0) for v in V)
    for s in V:
        S = []
        P = dict((w,[]) for w in V)
        g = dict((t, 0) for t in V); g[s] = 1
        d = dict((t,-1) for t in V); d[s] = 0
        Q = deque([])
        Q.append(s)
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in A[v]:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    g[w] = g[w] + g[v]
                    P[w].append(v)
        e = dict((v, 0) for v in V)
        while S:
            w = S.pop()
            for v in P[w]:
                e[v] = e[v] + (g[v]/g[w]) * (1 + e[w])
                if w != s:
                    C[w] = C[w] + e[w]
    return C











