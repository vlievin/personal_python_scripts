import sys
 
def inter( i ,j,  A ):

    ri = A.getrow(i)
    rj = A.getrow(j)

    l =  (ri * rj.transpose()).data
    if len(l):
        return (ri * rj.transpose()).data[0]
    else:
        return 0

def union( i,j, A ):
    return (A.getrow(i) + A.getrow(j)).nnz

def normJ( i , j, A ):
	if i != j:
		return  float(1.0 - float(inter(i,j,A))/float(union(i,j,A)))
	else:
		return 0
	

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
	
import time
import numpy as np

def DBSCAN( X, eps, minPts, norm ):
    
    

    C = 0
    t = time.time()
    m_last = 0
    clusters = []
    k = 0
    NOISE = []
    marked = []

    distances = np.zeros(( X.shape[0], X.shape[0] ) )
    index = []
    #compute distances
        
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[0]):
            index.append( [i,j] )
            
    def f( x):
#         if i!=j:
        return normJ(x[0],x[1],X)
#         else:
#             return 0    
                 
    dist_list =  map(f, index)

    for k in range(0,len(index)):
        distances[index[k][0], index[k][1]] = dist_list[k]
        distances[index[k][1], index[k][0]] = dist_list[k]
         
    print " distances have been computed in %s s" % str(time.time() - t)
                
    ids = range( 0 , X.shape[0] )
    ids_marked = []
    for i in ids:
        k += 1
        tt = time.time() - t
        m = int(tt) / 15
        if m != m_last:
            #print " time : " + str( tt ) + "  point : " + str( k ) +  "  clusters : " + str(len(clusters)) +"  m: " + str(m)
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
    print " loaded | n = " + str(n)
    print " --------------------------------------------------"
    #X = scipy.sparse.csr_matrix( X )
    t = time.time()
    clusters, noise = DBSCAN( X ,  eps, M, normJ)
    s = "output"
    s = s + str(n) + ".log"
    textFile = open(s , "w")
    textFile.write( "cost: " + str(time.time() - t) )
    textFile.write( "\n" )
	
    print " --------------------------------------------------"
    print "cost: " + str(time.time() - t)
    print
    print "nb clusters"
    textFile.write( "nb clusters : " +str(len(clusters)))
    textFile.write( "\n" )
    print len(clusters)
    print "length of each cluster"
    textFile.write( "length of each cluster" )
    textFile.write( "\n" ) 
    for c in clusters:
        print "   > " + str(len(c))
        textFile.write( "   > " + str(len(c)) )
        textFile.write( "\n" )
    print "noise qty"
    print len(noise)
    textFile.write( "noise qty" + str(len(noise)))
    textFile.write( "\n" )

    noise_exist = 0
    if len(noise):
        noise_exist = 1
    print "-------------------------------------------------- "
    print "nb clusters including noise" 
    print len(clusters) + noise_exist
    textFile.write("nb clusters including noise" + str(len(clusters) + noise_exist))
    print "-------------------------------------------------- "
    return
	
tryDBSCAN(int(sys.argv[1]), float(sys.argv[2]))
