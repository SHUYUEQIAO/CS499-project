##gradientDescent
GradientDescent <- function(X,y,stepSize,maxIterations)
{
  weightVector = matrix(0,nrow = ncol(X),ncol = 1)
  weightMatrix = matrix(0,nrow = ncol(X),ncol = 1)
  for(index in 1:maxIterations)
  {
    weightVector <- weightVector - stepSize * Grediant_Of_theta(X,y,weightVector)
    weightMatrix <- cbind(weightMatrix,weightVector)
  }
  
  weightMatrix <- weightMatrix[,-1]
  return(weightMatrix)
}

#grediant of theta
Grediant_Of_theta <- function(X,y,theta)
{
  grediantTheta <- matrix(0,ncol = ncol(X), nrow = 1)
  for( index in 1:nrow(X))
  {
    tempy <- y[index]
    tempX <- X[index,]
    yValue <- as.numeric(tempy)
    xValue <- as.numeric(tempX)
    expValue <- yValue%*%xValue%*%as.numeric(theta)
    upValue <- (-1)*yValue %*% xValue
    grediantThetaOfOne <- upValue / as.numeric(exp(expValue) + 1)
    grediantTheta <- grediantTheta + grediantThetaOfOne
  }
  
  grediantTheta <- grediantTheta / nrow(X)
  return(t(grediantTheta))
}