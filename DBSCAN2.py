def inter_preprocessing(x):
    return x* .5
    
def inter( i ,j,  A ):
#     s = 0
    ri = A.getrow(i)
    rj = A.getrow(j)
#     r = (ri + rj).getrow(0)
#     r.data[:] = inter_preprocessing(r.data)
#     print len(r.nonzero()[1])
#     print (ri * rj.transpose()).data[0]
    l =  (ri * rj.transpose()).data
    if len(l):
        return (ri * rj.transpose()).data[0]
    else:
        return 0
        
def union_bis( i,j, A ):
    s = 0
#     ri = A.getrow(i)
#     rj = A.getrow(j)
    for k in range(0 , A.shape[0]):
        if ( A[i, k ] or  A[j , k] ) > 0:
            s += 1
#     print "u"
#     print (ri + rj).nnz
#     print s
    return s

def union( i,j, A ):
    return (A.getrow(i) + A.getrow(j)).nnz

def normJ( i , j, A ):
    return  float(1.0 - float(inter(i,j,A))/float(union(i,j,A)))
	
def regionQuery( i , eps , D , distances, ids_marked):
    neighbors = []
    for j in range(0, D.shape[0]):
#         if j not in ids_marked:
        if distances[i,j] <= eps:
            neighbors.append(j)
            
    return neighbors
	
def expandNewCluster( ids, D, eps, minPts, distances, ids_marked):
    
    result = []
    j = 0
    while j < len(ids):
        i = ids[j]
        if i not in ids_marked:
            ids_marked.append(i)
            result.append( i )
            neighbors2 = regionQuery( i , eps, D, distances, ids_marked)
            if len(neighbors2) >= minPts:   
                for n in neighbors2:
                    if n not in ids_marked:
                        ids.append(n)
        j += 1
                    
    return result
	
from IPython.display import clear_output
import time
import numpy as np
import pathos.multiprocessing as mp

def DBSCAN( X, eps, minPts, norm ):
    C = 0
    t = time.time()
    m_last = 0
    clusters = []
    k = 0
    NOISE = []
    marked = []

    distances = np.empty(( X.shape[0], X.shape[0] ) )
    index = []
    #compute distances
    pool = mp.Pool(4)
    def f( x ):
        if i!=j:
            return normJ(x[0],x[1],X)
        else:
            return 0
        
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[0]):
            index.append( [i,j] )
            
    dist_list =  pool.map(f, index)
    for k in range(0,len(index)):
        distances[k[0, k[1]] ] = dist_list[k]
#             if i != j:
#                 d = norm(i,j,X)
#                 distances[i,j] = d
#                 distances[j,i] = d
#             else:
#                 distances[i,i] = 0
                
    print " distances have been computed in %s s" % str(time.time() - t)
                
    ids = range( 0 , X.shape[0] )
    ids_marked = []
    for i in ids:
        k += 1
        tt = time.time() - t
        m = int(tt) / 15
        if m != m_last:
            print " time : " + str( tt ) + "  point : " + str( k ) +  "  clusters : " + str(len(clusters)) +"  m: " + str(m)
            m_last = m
        if i not in ids_marked:
#             print len(marked)
            Neighbors = regionQuery( i , eps , X , distances, ids_marked)
            #print " n size : " + str(len(Neighbors))
            if (len(Neighbors) >=  minPts):
                C = expandNewCluster(  Neighbors, X , eps, minPts, distances, ids_marked)
                clusters.append( C )
#                 for p in C:
#                     ids_marked.append(p)
            else:           
                NOISE.append(i)
#                 print "++++++++++++++++++++++++++ "
            ids_marked.append( i )
            
            
    return clusters, NOISE
        
def tryDBSCAN( n = 10 , eps = 0.4):
    import cPickle as pickle
    import scipy
    import time
    
    M = 2
    X = pickle.load(open('data_%dpoints_%ddims.dat' %(n,n), 'rb'))
    print " --------------------------------------------------"
    print " loaded "
    print " --------------------------------------------------"
    #X = scipy.sparse.csr_matrix( X )
    t = time.time()
    clusters, noise = DBSCAN( X ,  eps, M, normJ)
    print " --------------------------------------------------"
    print "cost: " + str(time.time() - t)
    print
    print "nb clusters"
    print len(clusters)
    print "length of each cluster"
    for c in clusters:
        print "   > " + str(len(c))
    print "noise qty"
    print len(noise)

    noise_exist = 0
    if len(noise):
        noise_exist = 1
    print "-------------------------------------------------- "
    print "nb clusters including noise" 
    print len(clusters) + noise_exist
    print "-------------------------------------------------- "
    return
	
tryDBSCAN( 1000, 0.15 )
