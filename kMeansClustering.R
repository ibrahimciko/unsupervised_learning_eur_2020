####################################################
## Implementation of k-means cluster algorithm    ##
## see: Algorithm 14.1 HTF                        ##
####################################################
kMeans <- function(X,K, maxIter, nStart = 100, verbose = T){
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  n <- nrow(X)                                  #number of obs
  p <- ncol(X)                                  #number of features in X
  
  withinSS <- sum(dist(X))                      #initialize a very high withinSS
  algorithm <- 1                                #each algorithm corresponds to one K-means solution
  while (algorithm < nStart){                   #iterate nStart times to find lowest withinSS
    algorithm <- algorithm + 1
    distance <- matrix(NA, nrow = n, ncol = K)  #initialized distance to K clusters
    centers <- matrix(NA, nrow = K, ncol = p )  #initialized cluster centroids
    centerIndices <- sample(1:n,K,replace = F)  #sample K clusters from the observations
    
    for (i in seq_len(K)){
      centers[i,] <- X[centerIndices[i],]       #assign centroid coordinates as the observation coordinates
      distance[,i] <- sqrt(rowSums(sweep(X,2,centers[i,],"-")^2)) #for each centroid calculate distance to observations
    } 
    
    iter <- 0
    difference <- 10                             #initialize clusterMembership change for loop
    while (iter <= maxIter &  difference != 0)  {#loop until no change of memberships or maximumIter satisfied
      iter <- iter + 1
      
      memberships <- apply(distance,1, function(d) which(d == min(d))) #assign each observation to closest cluster
      
      for (i in seq_len(K)){                    
        if(sum(memberships == i) != 0) {
          centers[i,] <-  colMeans(X[which(memberships == i),,drop = F],2) # centroid update based on new memberships
          distance[,i] <- sqrt(rowSums(sweep(X,2,centers[i,],"-")^2))      # distance to Cluster calculation based on new centroids
        }
      }
      
      if (verbose) print(sum(apply(distance,1,min)))                        #calculate within sum of Squares
      updateMemberships <- apply(distance,1, function(d) which(d == min(d)))#assign to closest clusters after the update
      difference <- sum(memberships != updateMemberships)                   #sum total change between previous and new assignment
    }
    withinSSNew <- sum(apply(distance,1,min))                               #calculate within sum of squares
    if (withinSSNew < withinSS) {
    withinSS <- withinSSNew
    res <- list(withinSS,updateMemberships,centers,iter)
    names(res) <- c("withinSS","memberships","centers","iter")
    }
  }
  res
}
