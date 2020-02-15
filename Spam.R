#gradientDescent
GradientDescent <- function(X,y,stepSize,maxIterations)
{
  weightVector = matrix(0,nrow = ncol(X),ncol = 1)
  weightMatrix = matrix(0,nrow = ncol(X),ncol = maxIterations)
  for(index in 1:maxIterations)
  {
    weightVector <- weightVector - stepSize * Grediant_Of_theta(X,y,weightVector)
    weightMatrix[,index] <- weightVector[,1]
  }
  
  return(weightMatrix)
  #return(weightVector)
}

Grediant_Of_theta <- function(X,y,theta)
{
  m <- nrow(X)
  n <- ncol(X)
  oneMatrix <- matrix(1,nrow = m, ncol = 1)
  inExp <- y*as.matrix(X)%*%theta
  demonitor <- exp(inExp) + oneMatrix
  scale <- as.numeric(as.matrix(-y/demonitor)) 
  grediantMatrix <- X*scale
  grediantTheta <- colMeans(grediantMatrix)
  grediantTheta <- as.matrix(grediantTheta)
  
  return(grediantTheta)
}

Error_rate <- function(X,y,m,k)
{
  error_Matrix <- matrix(1:k, nrow = k, ncol = 1)
  testMatrix <- X*as.numeric(as.matrix(y))
  errValue <- as.matrix(colSums(testMatrix < 0))
  errRate <- errValue / m * 100
  error_Matrix <- cbind(errRate,error_Matrix)
  
  return(error_Matrix)
}

Error_rate_Value<- function(X,y,m,k)
{
  testMatrix <- X*as.numeric(as.matrix(y))
  errValue <- as.matrix(colSums(testMatrix < 0))
  errRate <- errValue / m * 100
  
  return(errRate)
}


GetMeanlogLoss_Matrix <- function(X,y,m,k)
{
  oneMatrix <- matrix(1,nrow = m, ncol = k)
  meanlogLoss_Matrix <- matrix(1:k, ncol = 1, nrow = k)
  inExp <- X*as.numeric(as.matrix(-y))
  inLog <- oneMatrix + exp(inExp)
  meanlogLoss <- colMeans(log(inLog))
  meanlogLoss_Matrix <- cbind(meanlogLoss, meanlogLoss_Matrix)
  
  return(meanlogLoss_Matrix)
}
#Experiment spam

# step 1:get the spam data

Data_No_Scale <- data.table::fread('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data')

#step 2:data processing



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

#get the most frequent for each data set
if(Training_Data_y_1 >= Training_Data_y_0)
{
  Frequent_Traning_Data <- matrix(1,ncol = 1, nrow = nrow(Training_Data_y))
  Frequent_Lable <- 1
}
if(Training_Data_y_1 < Training_Data_y_0)
{
  Frequent_Traning_Data <- matrix(-1,ncol = 1, nrow = nrow(Training_Data_y))
  Frequent_Lable <- -1
}
if(Frequent_Lable == 1)
{
  Frequent_Validation_Data <- matrix(1,ncol = 1, nrow = nrow(Validation_Data_y))
  Frequent_Testing_Data <- matrix(1,ncol = 1, nrow = nrow(Testing_Data_y))
}
if(Frequent_Lable == -1)
{
  Frequent_Validation_Data <- matrix(-1,ncol = 1, nrow = nrow(Validation_Data_y))
  Frequent_Testing_Data <- matrix(-1,ncol = 1, nrow = nrow(Testing_Data_y))
}

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

for( index in 1:nrow(Testing_Data_y))
{
  if(as.numeric(Testing_Data_y[index,]) == 0)
  {
    Testing_Data_y[index,] <- (-1)
  }
}

#compute matrix
Training_Weight_Matrix <- GradientDescent(Training_Data_X,Training_Data_y,
                                          0.2,1600)
Training_Predication_Matrix <- as.matrix(Training_Data_X) %*% as.matrix(Training_Weight_Matrix)

Validation_Predication_Matrix <- as.matrix(Validation_Data_X) %*% as.matrix(Training_Weight_Matrix)

Testing_Prediction_Matrix <- as.matrix(Testing_Data_X) %*% as.matrix(Training_Weight_Matrix)
k <- ncol(Training_Predication_Matrix)
m_Training <- nrow(Training_Predication_Matrix)
m_Validation <- nrow(Validation_Predication_Matrix)
m_Testing <- nrow(Testing_Prediction_Matrix)

#step 4: calculae the error rate and get graph

Validation_error_Matrix <- Error_rate(Validation_Predication_Matrix,Validation_Data_y,m_Validation,k)

colnames(Validation_error_Matrix) <- c('errorV','iteration')


Training_error_Matrix <- Error_rate(Training_Predication_Matrix,Training_Data_y,m_Training,k)

colnames(Training_error_Matrix) <- c('errorT','iteration')

errMatrix <- merge(Training_error_Matrix,Validation_error_Matrix,all = TRUE)


#find the min value of validation meanlogloss

Min_Validation_Matrix <- which(Validation_error_Matrix== min(Validation_error_Matrix[,1]),arr.ind = TRUE)

Min_Validation_Matrix_Col <- Min_Validation_Matrix[1]

#find the min value of training meanlogloss

Min_Training_Matrix <- which(Training_error_Matrix== min(Training_error_Matrix[,1]),arr.ind = TRUE)

Min_Training_Matrix_Col <- Min_Training_Matrix[1]

#get plot

Err_Data <- as.data.frame(errMatrix)

ggplot(data = Err_Data) +
  
  geom_line(mapping = aes(x = iteration, y = errorT, col = "train")) +
  
  geom_point(aes(Min_Training_Matrix_Col,as.numeric(Training_error_Matrix[Min_Training_Matrix_Col,1])))+
  
  geom_line(mapping = aes(x = iteration, y = errorV, col = "validation")) +
  
  geom_point(aes(Min_Validation_Matrix_Col,as.numeric(Validation_error_Matrix[Min_Validation_Matrix_Col,1])),col="red")+
  
  scale_colour_manual("",  breaks = c("validation", "train"),values = c("black", "red"))


#step 5: calculate the mean log loss and get the graph

#5.1 meanLog for training data

Training_MeanlogLoss_Matrix <- GetMeanlogLoss_Matrix(Training_Predication_Matrix,
                                                     
                                                     Training_Data_y,m_Training,k)

colnames(Training_MeanlogLoss_Matrix) <- c('meanlogLossT','iteration')



#5.2 meanLog for validiation data
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


#6 error rate
#6.1find the min theata
Min_loss_Iteration <- which(Validation_MeanlogLoss_Matrix== min(Validation_MeanlogLoss_Matrix),
                            arr.ind = TRUE) 
Min_Iteration_loss_Col <- Min_loss_Iteration[1]
Min_Iteration_loss_theta <- Training_Weight_Matrix[,Min_Iteration_loss_Col]
#log err for training
Min_Loss_Training <- as.matrix(Training_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
Loss_Training_Error <- Error_rate_Value(Min_Loss_Training,Training_Data_y,m_Training,k)
#log err for validation
Min_Loss_Validation <- as.matrix(Validation_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
Loss_Validation_Error <- Error_rate_Value(Min_Loss_Validation,Validation_Data_y,m_Validation,k)
#log err for testing
Min_Loss_Testing <- as.matrix(Testing_Data_X) %*% as.matrix(Min_Iteration_loss_theta)
Loss_Testing_Error <- Error_rate_Value(Min_Loss_Testing,Testing_Data_y,m_Testing,k)

#6.2 find the base line
Traning_Baseline_err <- Error_rate_Value(Frequent_Traning_Data,Training_Data_y,m_Training,k)
Validation_Baseline_err <- Error_rate_Value(Frequent_Validation_Data,Validation_Data_y,m_Validation,k)
Testing_Baseline_err <- Error_rate_Value(Frequent_Testing_Data,Testing_Data_y,m_Testing,k)

#6.3 create error table
Error_Table <- matrix(c(Loss_Training_Error,Loss_Validation_Error,Loss_Testing_Error,
                        Traning_Baseline_err,Validation_Baseline_err,Testing_Baseline_err),
                      nrow = 3, ncol = 2)
colnames(Error_Table) <- c('logLoss', 'Base')
rownames(Error_Table) <- c('Training', 'Validation','Testing')
Error_Table.table <- as.table(Error_Table)

#7 ROC curve
# compute default FPR and PTR
Default_Threshold_Matrix <- cbind(Min_Loss_Testing, Testing_Data_y)
tp <- 0
fp <- 0
fn <- 0
tn <- 0
if(Frequent_Lable == 1)
{
  for(index in 1:m_Testing)
  {
    Default_Threshold_Vector <- Default_Threshold_Matrix[index,]
    colnames(Default_Threshold_Vector) <- c('v1','v2')
    if(Default_Threshold_Vector$v1 > 0 & Default_Threshold_Vector$v2 == 1)
    {
      tp <- tp + 1
    }
    if(Default_Threshold_Vector$v1 < 0 & Default_Threshold_Vector$v2 == 1)
    {
      fn <- fn + 1
    }
    if(Default_Threshold_Vector$v1 > 0 & Default_Threshold_Vector$v2 == -1)
    {
      fp <- fp + 1
    }
    if(Default_Threshold_Vector$v1 < 0 & Default_Threshold_Vector$v2 == -1)
    {
      tn <- tn + 1
    }
  }
  #TPR=(TP/(TP+FP)) and FPR =(FP/(FP+TN))
  tpr <- (tp/(tp + fp))
  fpr <- (fp/(fp + tn))
}

if(Frequent_Lable == -1)
{
  for(index in 1:m_Testing)
  {
    Default_Threshold_Vector <- Default_Threshold_Matrix[index,]
    colnames(Default_Threshold_Vector) <- c('v1','v2')
    if(Default_Threshold_Vector$v1 < 0 & Default_Threshold_Vector$v2 == -1)
    {
      tp <- tp + 1
    }
    if(Default_Threshold_Vector$v1 > 0 & Default_Threshold_Vector$v2 == -1)
    {
      fn <- fn + 1
    }
    if(Default_Threshold_Vector$v1 < 0 & Default_Threshold_Vector$v2 == 1)
    {
      fp <- fp + 1
    }
    if(Default_Threshold_Vector$v1 > 0 & Default_Threshold_Vector$v2 == 1)
    {
      tn <- tn + 1
    }
  }
  #TPR=(TP/(TP+FP)) and FPR =(FP/(FP+TN))
  tpr <- (tp/(tp + fp))
  fpr <- (fp/(fn + tn))
}

#model for logloss
Logloss_Prediction <- Min_Loss_Testing
pred = prediction(Logloss_Prediction, Testing_Data_y)
roc = performance(pred,"tpr","fpr")
plot(roc, col="red")
points(0,0)
points(fpr,tpr,col="red")
abline(a = 0, b = 1) 
legend("bottomright",inset = 0.01, c("logistic regression","baseline"),
       col=c("red","black"),lty=c(1),pch=c(1))

