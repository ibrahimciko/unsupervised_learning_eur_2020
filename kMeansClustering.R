####################################################
## Implementation of k-means cluster algorithm    ##
## see: Algorithm 14.1 HTF                        ##
####################################################
kMeans <- function(X,K, maxIter, nStart = 100, verbose = T){
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  n <- nrow(X) #number of obs
  p <- ncol(X) #number of features in X
  
  withinSS <- sum(dist(X)) #initialize a very high withinSS
  algorithm <- 1   #each algorithm corresponds to one K-means solution
  while (algorithm < nStart){ #iterate nStart times to find lowest withinSS
    algorithm <- algorithm + 1
    distance <- matrix(NA, nrow = n, ncol = K) #initialized distance to K clusters
    centers <- matrix(NA, nrow = K, ncol = p ) #initialized cluster centroids
    centerIndices <- sample(1:n,K,replace = F) #sample K clusters from the observations
    
    for (i in seq_len(K)){
      centers[i,] <- X[centerIndices[i],] #assign centroid coordinates as the observation coordinates
      distance[,i] <- rowSums(sweep(X,2,centers[i,],"-")^2) #for each centroid calculate distance to observations
    } 
    
    iter <- 0
    difference <- 10 #initialize clusterMembership change for loop
    while (iter <= maxIter &  difference != 0){#loop until no change of memberships or maximumIter satisfied
      iter <- iter + 1
      
      memberships <- apply(distance,1, function(d) which(d == min(d))) #assign each observation to closest cluster
      
      for (i in seq_len(K)){                    
        if(sum(memberships == i) != 0) {
          centers[i,] <-  colMeans(X[which(memberships == i),,drop = F],2) #centroid update based on new memberships
          distance[,i] <- rowSums(sweep(X,2,centers[i,],"-")^2) #distance to Cluster calculation based on new centroids
        }
      }
      
      if (verbose) print(sum(apply(distance,1,min))) #calculate within sum of Squares
      updateMemberships <- apply(distance,1, function(d) which(d == min(d)))#assign to closest clusters after the update
      difference <- sum(memberships != updateMemberships) #sum total change between previous and new assignment
    }
    withinSSNew <- sum(apply(distance,1,min)) #calculate within sum of squares
    if (withinSSNew < withinSS) {
      withinSS <- withinSSNew
      res <- list(withinSS,updateMemberships,centers,iter)
      names(res) <- c("withinSS","memberships","centers","iter")
    }
  }
  res
}

############ COMPARISON WITH kmeans() FUNCTION ##############
df <- read.csv("cats_vs_dogs.csv") #load the data
X  <- subset(df, select = -c(X,state)) #remove the states and X the index variable
X<- scale(X)

withinErrors <- matrix(ncol = 3, nrow = nrow(X) - 1,dimnames = list(NULL,c("k","ourAlgorithm","kMeansPackage")))
withinErrors[,1] <- 1: (nrow(X) - 1)
for (i in seq_len(nrow(X - 2))) {
  withinErrors[i,2] <- kMeans(X = X, K =  i, maxIter = 100, nStart = 500,verbose = F)$withinSS
  withinErrors[i,3] <- kmeans(X,centers = i ,nstart = 100)$tot.withinss
}
ggplot()+
  geom_line(data = data.frame(withinErrors), 
            aes(y = ourAlgorithm,x = k,colour = "darkblue"),size = 1, alpha = 1)+
  geom_line(data = data.frame(withinErrors),
            aes(y = kMeansPackage,x = k,colour = "re"),size = 1, linetype = "longdash")+
  scale_color_discrete(name = "Programs", labels = c("our package", "kmeans()"))+ labs(y = "Within SS") + 
  annotate( geom = "text", x = 33, y = 500, label = "500 random initial starts")
