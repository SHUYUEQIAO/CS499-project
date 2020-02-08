##gradientDescent
GradientDescent <- function(X,y,stepSize,maxIterations)
{
  weightVector = matrix(0,nrow = ncol(X),ncol = 1)
  weightMatrix = matrix(0,nrow = ncol(X),ncol = 1)
  for(index in 1:maxIterations)
  {
    weightVector <- weightVector - stepSize * Grediant_Of_theta(X,y,weightVector)
    #weightMatrix[,index] <- weightVector[,1]
    weightMatrix <- cbind(weightMatrix,weightVector)
  }
  
  weightMatrix <- weightMatrix[,-1]
  return(weightMatrix)
  #return(weightVector)
}

#grediant of theta
Grediant_Of_theta <- function(X,y,theta)
{
  grediantTheta <- matrix(0,ncol = ncol(X), nrow = 1)
  for( index in 1:nrow(X))
  {
    #exp(as.numeric(Training_Data_y[2,])%*%
    #as.numeric(Training_Data_X[3,])%*%as.numeric(weightVector))
    #up = (-1)*as.numeric(Training_Data_y[2,])%*%as.numeric(Training_Data_X[1,])
    #up/numeric(exp)
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

# error rate
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

#Experiment spam

# step 1:get the spam data
Data_No_Scale <- data.table::fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/SAheart.data')

#step 2:data processing
for (index_row in 1:nrow(Data_No_Scale)){
  if(as.character(Data_No_Scale[index_row,6]) == 'Present')
  {
    Data_No_Scale[index_row,6] <- 1
  }
  else
  {
    Data_No_Scale[index_row,6] <- 0
  }
}
  Data_No_Scale$famhist <- as.numeric(Data_No_Scale$famhist)
  
  #scale the inputs
  m <- nrow(Data_No_Scale)
  n <- ncol(Data_No_Scale)
  Data_No_Scale_X <- Data_No_Scale[,-..n]
  Data_No_Scale_y <- Data_No_Scale[,..n]
  Data_Scale_X <- scale(Data_No_Scale_X, TRUE, TRUE)
  Data <- cbind(Data_Scale_X,Data_No_Scale_y)
  m <- nrow(Data)
  n <- ncol(Data)
  
  #randomly split the data
  Training_Sample_size = 0.4*m
  Picked_Training <- sample(seq_len(m),size = Training_Sample_size)
  Other_Data <- Data[Picked_Training,]
  Training_Data <- Data[-Picked_Training,]
  
  m_other <- 0.5*nrow(Other_Data)
  Validation_Sample_size <- m_other
  Picked_Validation <- sample(seq_len(m_other),size = Validation_Sample_size)
  Validation_Data <- Other_Data[Picked_Validation,]
  Testing_Data <- Other_Data[-Picked_Validation,]
  
  
  Training_Data_X <- Training_Data[,-..n]
  Training_Data_y <- Training_Data[,..n]
  colnames(Training_Data_y) <- 'V1'
  Validation_Data_X <- Validation_Data[,-..n]
  Validation_Data_y <- Validation_Data[,..n]
  colnames(Validation_Data_y) <- 'V1'
  Testing_Data_X <- Testing_Data[,-..n]
  Testing_Data_y <- Testing_Data[,..n]
  colnames(Testing_Data_y) <- 'V1'
  
  #create a table of counts with a row for each set
  #(train/validation/test) and a column for each class (0/1)
  Training_Data_y_1 <- length(which(Training_Data_y$V1 == 1))
  Training_Data_y_0 <- length(which(Training_Data_y$V1 == 0))
  Validation_Data_y_1 <- length(which(Validation_Data_y$V1 == 1))
  Validation_Data_y_0 <- length(which(Validation_Data_y$V1 == 0))
  Testing_Data_y_1 <- length(which(Testing_Data_y$V1 == 1))
  Testing_Data_y_0 <- length(which(Testing_Data_y$V1 == 0))
  
  Divid_Table <- matrix(c(Training_Data_y_1,Validation_Data_y_1,Testing_Data_y_1,
                          Training_Data_y_0,Validation_Data_y_0,Testing_Data_y_0),
                        nrow=3,
                        ncol=2)
  colnames(Divid_Table) <- c('1', '0')
  rownames(Divid_Table) <- c('Training', 'Validation','Testing')
  Divid_Table.table <- as.table(Divid_Table)
  
  #step 3:compute a learned weightMatrix, training predicated matrix, vaildation predicated matrix
  #transfer y to {-1,1}
  for( index in 1:nrow(Training_Data_X))
  {
    if(as.numeric(Training_Data_y[index,]) == 0)
    {
      Training_Data_y[index,] <- (-1)
    }
  }
  for( index in 1:nrow(Validation_Data_X))
  {
    if(as.numeric(Validation_Data_y[index,]) == 0)
    {
      Validation_Data_y[index,] <- (-1)
    }
  }
  
  #get predicated value
  Training_Weight_Matrix <- GradientDescent(Training_Data_X,Training_Data_y,
                                            0.5,300)
  Training_Predication_Matrix <- as.matrix(Training_Data_X) %*% as.matrix(Training_Weight_Matrix)
  Validation_Predication_Matrix <- as.matrix(Validation_Data_X) %*% as.matrix(Training_Weight_Matrix)
  k <- ncol(Training_Predication_Matrix)
  
  #step 4: calculae the error rate and get graph
  Validation_error_Matrix <- Error_rate(Validation_Predication_Matrix,Validation_Data_y)
  colnames(Validation_error_Matrix) <- c('errorV','iteration')
  
  Training_error_Matrix <- Error_rate(Training_Predication_Matrix,Training_Data_y)
  colnames(Training_error_Matrix) <- c('errorT','iteration')
  errMatrix <- merge(Training_error_Matrix,Validation_error_Matrix,all = TRUE)
  #find the min value of validation meanlogloss
  Min_Validation_Matrix <- which(Validation_error_Matrix[,1] == min(Validation_error_Matrix[,1]),arr.ind = TRUE)
  Min_Validation_Matrix_Col <- Min_Validation_Matrix[1]
  #find the min value of training meanlogloss
  Min_Training_Matrix <- which(Training_error_Matrix[,1]== min(Training_error_Matrix[,1]),arr.ind = TRUE)
  Min_Training_Matrix_Col <- Min_Training_Matrix[1]
  
  
  #get plot
  Err_Data <- as.data.frame(errMatrix)
  ggplot(data = Err_Data) +
    geom_line(mapping = aes(x = iteration, y = errorT, col = "train")) +
    geom_point(aes(Min_Training_Matrix_Col,as.numeric(Training_error_Matrix[Min_Training_Matrix_Col,1])))+
    geom_line(mapping = aes(x = iteration, y = errorV, col = "validation")) +
    geom_point(aes(Min_Validation_Matrix_Col,
                   as.numeric(Validation_error_Matrix[Min_Validation_Matrix_Col,1])),col="red")+
    scale_colour_manual("",  breaks = c("validation", "train"),values = c("black", "red"))
  
  
  
  #step 5: calculate the mean log loss and get the graph
  #5.1 meanLog for training data
  m_Training <- nrow(Training_Predication_Matrix)
  Training_MeanlogLoss_Matrix <- GetMeanlogLoss_Matrix(Training_Predication_Matrix,
                                                       Training_Data_y,m_Training,k)
  colnames(Training_MeanlogLoss_Matrix) <- c('meanlogLossT','iteration')
  
  #5.2 meanLog for validiation data
  m_Validation <- nrow(Validation_Predication_Matrix)
  Validation_MeanlogLoss_Matrix <- GetMeanlogLoss_Matrix(Validation_Predication_Matrix,
                                                         Validation_Data_y,m_Validation,k)
  colnames(Validation_MeanlogLoss_Matrix) <- c('meanlogLossV','iteration')
  
  #find the min value of validation meanlogloss
  Min_Validation_Matrix <- which(Validation_MeanlogLoss_Matrix== min(Validation_MeanlogLoss_Matrix),arr.ind = TRUE)
  Min_Validation_Matrix_Col <- Min_Validation_Matrix[1]
  #find the min value of training meanlogloss
  Min_Training_Matrix  <- which(Training_MeanlogLoss_Matrix== min(Training_MeanlogLoss_Matrix),arr.ind = TRUE)
  Min_Training_Matrix_Col <- Min_Training_Matrix[1]
  
  #5.3 get plot
  loss_Matrix <- merge(Training_MeanlogLoss_Matrix,Validation_MeanlogLoss_Matrix,all = TRUE)
    ggplot(data = as.data.frame(loss_Matrix)) +
    geom_line(mapping = aes(x = iteration, y = meanlogLossT,col = "train")) +
    geom_point(aes(Min_Training_Matrix_Col,as.numeric(Training_MeanlogLoss_Matrix[Min_Training_Matrix_Col,1])))+
    geom_line(mapping = aes(x = iteration, y = meanlogLossV, col = "validation"))+
    geom_point(aes(Min_Validation_Matrix_Col,as.numeric(Validation_MeanlogLoss_Matrix[Min_Validation_Matrix_Col,1])),col="red")+
    scale_colour_manual("",  breaks = c("validation", "train"),values = c("black", "red"))
  
  Min_loss_Iteration <- which(Validation_MeanlogLoss_Matrix== min(Validation_MeanlogLoss_Matrix),
                              
                              arr.ind = TRUE) 
  
  Min_Iteration_loss_Col <- Min_loss_Iteration[1]
  
  Min_Iteration_loss_theta <- Training_Weight_Matrix[,Min_Iteration_loss_Col]
  
  Min_Loss_Training <- as.matrix(Training_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
  
  Loss_Training_Error_Matrix <- Error_rate(Min_Loss_Training,Training_Data_y)
  
  Loss_Training_Error <- as.numeric(Loss_Training_Error_Matrix[,1])
  
  Min_Loss_Validation <- as.matrix(Validation_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
  
  Loss_Validation_Error_Matrix <- Error_rate(Min_Loss_Validation,Validation_Data_y)
  
  Loss_Validation_Error <- as.numeric(Loss_Validation_Error_Matrix[1,1])
  
  
  
  Min_Loss_Testing <- as.matrix(Testing_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
  
  Loss_Testing_Error_Matrix <- Error_rate(Min_Loss_Testing,Testing_Data_y)
  
  Loss_Testing_Error <- as.numeric(Loss_Testing_Error_Matrix[,1])
  
  
  
  Min_Base_Iteration <- which(Training_error_Matrix[,1]== min(Training_error_Matrix[,1]),
                              
                              arr.ind = TRUE)
  
  Min_Iteration_Base_col <- Min_Base_Iteration[1]
  
  Min_Iteration_Base_theta <- Training_Weight_Matrix[,Min_Iteration_Base_col]
  
  Min_Base_Training <- as.matrix(Training_Data_X) %*% as.matrix(Min_Iteration_Base_theta)
  
  Base_Training_Error_Matrix <- Error_rate(Min_Base_Training,Training_Data_y)
  
  Base_Training_Error <- as.numeric(Loss_Training_Error_Matrix[,1])
  
  
  
  Min_Base_Validation <- as.matrix(Validation_Data_X) %*% as.matrix(Min_Iteration_Base_theta)
  
  Base_Validation_Error_Matrix <- Error_rate(Min_Base_Validation,Validation_Data_y)
  
  Base_Validation_Error <- as.numeric(Base_Validation_Error_Matrix[1,1])
  
  
  
  Min_Base_Testing <- as.matrix(Testing_Data_X) %*% as.matrix(Min_Iteration_Base_theta)
  
  Base_Testing_Error_Matrix <- Error_rate(Min_Base_Testing,Testing_Data_y)
  
  Base_Testing_Error <- as.numeric(Base_Testing_Error_Matrix[,1])
  
  #get table
  
  Error_Table <- matrix(c(Loss_Training_Error,Loss_Validation_Error,Loss_Testing_Error,
                          
                          Base_Training_Error,Base_Validation_Error,Base_Testing_Error),
                        
                        nrow=3,
                        
                        ncol=2)
  
  colnames(Error_Table) <- c('logLoss', 'Base')
  
  rownames(Error_Table) <- c('Training', 'Validation','Testing')
  
  Error_Table.table <- as.table(Error_Table)