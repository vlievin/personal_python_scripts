import cPickle as pickle
import scipy
import time

#def norm

def inter_preprocessing(x):
    return x* .5
    
def inter( i ,j,  A ):
    s = 0
    ri = A.getrow(i)
    rj = A.getrow(j)
    r = (ri + rj).getrow(0)
    r.data[:] = inter_preprocessing(r.data)
    return len(r.nonzero()[1])
        
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
	
	
#def regionQuery

def regionQuery( i , eps , D , norm, ids_marked):
    neighbors = []
    for j in range(0, D.shape[0]):
#         if j not in ids_marked:
        if norm(i, j, D) <= eps:
            neighbors.append(j)
            
    return neighbors
	
#def expandNewCluster
def expandNewCluster( ids, D, eps, minPts, norm, ids_marked):
    
    result = []
    j = 0
    while j < len(ids):
        i = ids[j]
        if i not in ids_marked:
            ids_marked.append(i)
            result.append( i )
            neighbors2 = regionQuery( i , eps, D, norm, ids_marked)
            if len(neighbors2) >= minPts:   
                for n in neighbors2:
                    if n not in ids_marked:
                        ids.append(n)
        j += 1
                    
    return result
	
	
def DBSCAN( X, eps, minPts, norm ):
    C = 0
    t = time.time()
    m_last = 0
    clusters = []
    k = 0
    NOISE = []
    marked = []
    ids_to_process = range( 0 , X.shape[0] )
    points_to_proces = X
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
            Neighbors = regionQuery( i , eps , X , norm, ids_marked)
            #print " n size : " + str(len(Neighbors))
            if (len(Neighbors) >=  minPts):
#                 print " : "
                #points_to_proces = [ X[k,:] for k in ids_to_process]
                C = expandNewCluster(  Neighbors, X , eps, minPts, norm, ids_marked)
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
	s = "output"
	s = s + str(n)
	textFile = open(s , "w")
	textFile.write( "cost: " + str(time.time() - t) )
	
    print " --------------------------------------------------"
    print "cost: " + str(time.time() - t)
    print
    print "nb clusters"
	textFile.write( "nb clusters : " +str(len(clusters)))
    print len(clusters)
    print "length of each cluster"
	textFile.write( "length of each cluster" )
    for c in clusters:
        print "   > " + str(len(c))
		textFile.write( "   > " + str(len(c)) )
    print "noise qty"
    print len(noise)
	textFile.write( "noise qty" + str(len(noise)))

    noise_exist = 0
    if len(noise):
        noise_exist = 1
    print "-------------------------------------------------- "
    print "nb clusters including noise" 
    print len(clusters) + noise_exist
	textFile.write("nb clusters including noise" :  str(len(clusters) + noise_exist))
    print "-------------------------------------------------- "
    return


tryDBSCAN()
	

	

	
	


	
