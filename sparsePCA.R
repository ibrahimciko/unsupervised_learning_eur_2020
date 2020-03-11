initializeV <- function(X){ 
  # Function to perform rank 1 SVD. It obtains v1 to initialize sparse PCA algorithm
  # Inputs:
  #   X: Data Matrix (Mean-Centered)
  # Outputs:
  #   v_init: First eigenvector of the Covariance Matrix of X via EigenDecomposition method
  
  S <- t(X) %*% X / (dim(X)[1]-1)
  eigenDecomp <- eigen(S,symmetric = T)
  v_init <- eigenDecomp$vectors[,1]
  return(v_init)
}

myNorm <- function(X,d){
  # Function to calculate norm of the Matrix of degree d. That is, ||X||_d ^(1/d)
  # Inputs:
  #   X: Data Matrix
  #   d: degree of the norm
  # Outputs:
  #   res: Norm of the Matrix
  
  if (d == 1){
    res <- sum(abs(X))
  } else {
    res<- sum(X^d)^(1/d)
  }
  res
}

plus <- function(X){
  # Function to modify all the nonnegatives as 0 (+ operator in the paper by Tibsherani)
  # Inputs:
  #   X: Data Matrix
  # Outputs:
  #   X: Modified X
  
  positive_mask <- X > 0
  X[!positive_mask] <- 0
  return(X)
}

softThresh <- function(X,lambda){
  # Function to do soft thresholding.
  # Inputs:
  #   X: Data Matrix
  #   lambda: when 0, function just returns the input itself without any modification
  # Outputs:
  #   S: Modified X
  
  S <- sign(X) * plus((abs(X) - lambda)) 
  return(S)
}


updateU <- function(X,v,lambda1){ 
  # The updating function of the left singular vector u
  # Inputs:
  #   X: Data Matrix
  #   v: Left singular vector
  #   lambda1: soft thresholding scalar
  # Outputs:
  #   u: updated right singular vector  
  #   norm: norm of u
  
  xProdV <- as.vector(X %*% v)
  S <- softThresh(xProdV,lambda1)
  u <- S /(myNorm(xProdV,2)+ 1e-10)
  norm <- myNorm(u,1)
  
  result <- list(u,norm)
  names(result) <- c("u","norm")
  return(result)
}


updateV <- function(X,u,lambda2){
  # The updating function of the right singular vector v
  # Inputs:
  #   X: Data Matrix
  #   u: Left singular vector
  #   lambda2: soft thresholding scalar
  # Outputs:
  #   v: updated right singular vector  
  #   norm: norm of v
  
  xProdU <- as.vector(t(X) %*% u )
  S <- softThresh(xProdU,lambda2)
  v <- S/(myNorm(S,2)+1e-10)
  norm <- myNorm(v,1) # d for the penalty
  
  result <- list(v,norm)
  names(result) <- c("v","norm")
  return(result)
}

binarySearch <- function(FUN,X,vector,min =0,max = 5000,target,tol){
  # A recursive binary search function to find the optimal value of lambda given target.
  # It finds the smallest lambda such that norm of the singular vector equals to target (c1 or c2)
  # Inputs:
  #   FUN: function in which binarySearch is used
  #   X: Data Matrix
  #   vector: Left or right singular vector, depending on which we are updating
  #   min: minimum lambda value for soft-thresholding
  #   max: maximum lambda value for soft-thresholdign
  #   target: the optimization constraint. norm(vector) <= target
  #   tol: stopping criterion. If the difference between the norm of the lambda and the target is
  #        smaller than tol, stop the algorithm
  # Outputs:
  #   optimalPenalty: optimal thresholding scalar 
  #   norm: norm of the singular vector given the optimalPenalty
  
  updateMin <- FUN(X,vector,min)
  stopifnot(updateMin$norm > target)
  updateMax <- FUN(X,vector,max)
  stopifnot(updateMax$norm < target)
  middle <- (min + max) / 2
  updateMiddle <- FUN(X,vector,middle)
  
  if (abs(updateMiddle$norm - target) < tol){
    result <- list(middle,updateMiddle$norm)
    names(result) <- c("optimalPenalty","norm")
    return(result)
    
  } else if (updateMiddle$norm > target) {
    min <- middle
    return(binarySearch(FUN,X,vector,min,max,target,tol))
    
  } else {
    max <- middle
    return(binarySearch(FUN,X,vector,min,max,target,tol))
  }
}

sparseePCA1 <- function(X,scale = F,center = F,c1 = 10000,c2 = 10000,tol){
  # Main function to calculate rank1 singular vectors u,v and the singular value sigma.
  # Inputs:
  #   X: Data Matrix
  #   scale: If TRUE, standardize the data Matrix
  #   center: If True, mean-centers the data Matrix
  #   c1: L1 regularization parameter on u
  #   c2: L1 regularization parameter on v
  #   tol: stopping criterion for convergence
  
  # Outputs:
  #   u: left singular vector
  #   sigma: corresponding singular value
  #   v: right singular vector
  
  X <- as.matrix(X) 
  
  if (scale == T){
    X <- scale(X, center =  F)
  }
  
  if (center == T) {
    X <- scale(X, scale = F)
  }
  
  v <- initializeV(X)
  u <- rnorm(n = dim(X)[1]) 
  
  iter <- 1
  maxIter <- 100
  convergeU <- FALSE
  convergeV <- FALSE
  while( (iter < maxIter) & (!convergeU & !convergeV) ){
    iter <- iter + 1
    #update U
    uRes <- updateU(X,v,0)
    uNorm <- uRes$norm
    if (uNorm <= c1) {
      uCandid <- uRes$u
    } else if (uNorm > c1){
      resBinary <- binarySearch(updateU,X,v,0,5000,c1,tol)
      optimalPenalty <- resBinary$optimalPenalty
      stopifnot(optimalPenalty > 0)
      uCandid <- updateU(X,v,optimalPenalty)$u
    }
    convergeU <- ( myNorm((u - uCandid),2) < tol)
    u <- uCandid
    
    #update V
    vRes <- updateV(X,u,0)
    vNorm <- vRes$norm
    if (vNorm <= c2) {
      vCandid <- vRes$v
    } else if (vNorm > c2){
      resBinary <- binarySearch(updateV,X,u,0,5000,c2,tol)
      optimalPenalty <- resBinary$optimalPenalty
      stopifnot(optimalPenalty > 0)
      vCandid <- updateV(X,u,optimalPenalty)$v
    }
    convergeV <- ( myNorm(v - vCandid,2) < tol)
    v <- vCandid
  }
  #eigenvector
  sigma <- drop(t(u) %*% X %*% v)
  
  #results
  res <- list(u, sigma, v)
  names(res) <- c("u","sigma","v")
  return(res)
}

sparsePCA <- function(FUN,X, scale = F, center = T, c1 =1000, c2 = 1000, tol =0.01,r){
  # Main function to calculate rank r left and right singular vectors u,v
  # Inputs:
  #   FUN: The function that calculates rank1 singular vectors (see above)
  #   X: Data Matrix
  #   scale: If TRUE, standardize the data Matrix
  #   center: If True, mean-centers the data Matrix
  #   c1: L1 regularization parameter on u
  #   c2: L1 regularization parameter on v
  #   tol: stopping criterion for convergence
  #   r: Rank of the matrix V, or number of eigenvectors
  
  # Outputs:
  #   U: The matrix of  left singular vectors, Rank: r
  #   sigma: Diagonal matrix of first r singular values
  #   v: the matrix of  right singular vector, Rank: r
  
  V <- matrix(NA,nrow = dim(X)[2], ncol = r)
  U <- matrix (NA,nrow = dim(X)[1],ncol = r)
  sigma <- vector(mode = "double", length = r)
  for (i in seq_len(r)){
    res <- FUN(X,scale,center,c1,c2,tol)
    u <- res$u;     U[,i] <- u;
    d <- res$sigma; sigma[i] <- d
    v <- res$v;     V[,i] <- v
    X <- X - d * u %*% t(v)
  }
  if (r == 1) {
    XConstruct <- sigma * U %*% t(V)
  } else {
    XConstruct <-  U %*% diag(sigma) %*% t(V)
  }
  
  result <- list(V,U,sigma, XConstruct)
  names(result) <- c("V","U","Sigma","XConstruct")
  return(result)
}

#Example
library(PMA)
load("FIFA2017_NL.RData") #data
skills <- fifa[,4:(ncol(fifa)- 3)]  #exclude money and team and player name
skills <- scale(skills, scale  = F) #mean center

SparcePcaResults <- matrix(,nrow = 29, ncol = )
for (i in seq_len(dim(skills)[2])){
  res <- sparsePCA(sparsePCA1,skills,F,F,c1 = 1000, c2 = 2, tol = 1e-5, i)
  X <- res$XConstruct
  SparcePcaResults[i,1] <- myNorm(X-skills,2) 
  SparcePcaResults[i,2] <-sum(abs(res$U-0) < 1e-5)
  SparcePcaResults[i,3] <-sum(abs(res$V-0) < 1e-5)
}
colnames(SparcePcaResults) <- c("error", "U0", "V0")

SPCResults <- matrix(,nrow = 29, ncol = 3)
for (i in seq_len(dim(skills)[2])){
  res <- SPC(skills,sumabsv = 2, K = i, center = F)
  u <- res$u
  v <- res$v
  sigma <- res$d
  if (length(sigma) == 1) {
    X <- sigma * u %*% t(v) 
  } else {
    X <- u %*% diag(sigma) %*% t(v)
  }
  SPCResults[i,1] <- myNorm(X - skills,2)
  SPCResults[i,2] <- sum(abs(u-0) < 1e-5)
  SPCResults[i,3] <- sum(abs(v-0) < 1e-5)
}
colnames(SPCResults) <- c("error", "U0", "V0")

#plotting
df <- data.frame(cbind(1:29,SparcePcaResults[,"error"],SPCResults[,"error"]))
colnames(df) <- c("x","errorOurFunction","errorSPC")
#plot
library(ggplot2)
ggplot() + 
  geom_line(data = df,aes(y = errorOurFunction, x = x,  color = "darkred"),size = 1) + 
  geom_line(data = df,aes(y = errorSPC, x = x,color="steelblue"), size = 1 ) +
  labs(x = "rank", y = "Reconstruction Error") +
  scale_color_discrete(name = "Reconstruction Errors", labels = c("our implementation", "package"))

df_missing <- data.frame(cbind(1:29,SparcePcaResults[,"V0"],SPCResults[,"V0"]))
colnames(df_missing) <- c("rank","v0ours","v0package")

ggplot() + 
  geom_line(data = df_missing,aes(y = v0ours , x = rank,  color = "darkred"),size = 1) + 
  geom_line(data = df_missing,aes(y = v0package, x = rank,color="steelblue"), size = 1 ) +
  labs(x = "rank", y = "#of zero loadings") +
  scale_color_discrete(name = "Zero Loadings", labels = c("our implementation", "package"))