Error_rate <- function(Matrix,Matrix2)
{
  Validation_error_Matrix <- matrix(0,nrow = ncol(Matrix), ncol = 2)
  for (index_col in 1:ncol(Matrix))
  {
    error <- 0
    right <- 0
    for (index_row in 1:nrow(Matrix))
    {
      if((as.numeric(Matrix[index_row,index_col])>0 & as.numeric(Matrix2[index_row,])==1)
         ||(as.numeric(Matrix[index_row,index_col])<0 & as.numeric(Matrix2[index_row,])==-1))
        
      {
        right <- right+1
      }
      else
      {
        error <- error+1
      }
    }
    Validation_error_Matrix[index_col,1] <- error/(error+right)*100
    Validation_error_Matrix[index_col,2] <- index_col
  }
  return(Validation_error_Matrix)
}

#calcualte meanlogLoss
GetMeanlogLoss_Matrix <- function(X,y,m,k){
  
  meanlogLoss_Matrix <- matrix(0, ncol = 2, nrow = k)
  for(index_n in 1:k)
  {
    logLoss <- 0
    for(index_m in 1:m){
      
      ExpValue <- (-1)*as.numeric(y[index_m,])%*%
        as.numeric(X[index_m,index_n])
      
      OnelogLoss <- log(as.numeric(exp(ExpValue)) + 1)
      
      logLoss <- OnelogLoss + logLoss
      
    }
    meanlogLoss <- logLoss / m
    
    meanlogLoss_Matrix[index_n,1] <- meanlogLoss
    meanlogLoss_Matrix[index_n,2] <- index_n
  }
  return(meanlogLoss_Matrix)
}