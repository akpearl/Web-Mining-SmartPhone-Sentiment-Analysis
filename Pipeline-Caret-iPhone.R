#
# Project name: overal sentiment toward iPhone (data collected fom the web using AWS)
# File name:    iPhonefile (70% - train and 30% -test ds) 
##############################################################
# Housekeeping
##############################################################

rm(list = ls()) # Clear objects if necessary
getwd()         # find out where the working directory is currently set
                # set working directory 
setwd("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Web Mining")

##############################################################
# Load packages
##############################################################
library(doParallel)  #parallel processing for faster execution
library(plyr)   # for revalue function 
install.packages("party") 
install.packages("caret")
install.packages("corrplot")
install.packages("randomForest") 
install.packages("C50")
install.packages("rattle", repos="http://rattle.togaware.com", type="source") 
#install.packages("rattle")  for graphics
install.packages("arules")
#install.packages("xgboost")

require(party)
require(caret)  
require(corrplot)
require(randomForest)
require(C50)
require(rattle)
require(lattice)
require(arules)
require(doParallel)
require(plyr)  
#require(xgboost)

#the following lines will create a local 4-node snow cluster
workers=makeCluster(4,type="SOCK")
registerDoParallel(workers)
foreach(i=1:4) %dopar% Sys.getpid()
############################################################
# Import data
############################################################

#read input data file
iPhonefile <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Web Mining/Output Files/Wet100Run/iPhoneLargeMatrix.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)

# view data structure                
str(iPhonefile)  # 13,741 obs of 60 vars 

# save a copy of input file in working directory before any changes made
write.csv(iPhonefile, "iPhonefile_sentimet.csv")

#######################################
# Evaluate data
####################################### 

head(iPhonefile,5)
tail(iPhonefile,5)
summary(iPhonefile)

options(max.print=1000000) # print default is only 20 vars

histogram(iPhonefile$iphoneSentiment, data = NULL,plot=F)
#correlations <- cor(iPhonefile[,c(1-60)]) 
correlations <- cor(iPhonefile)
corrplot(correlations, method = "circle")
corrplot(correlations, order ="hclust")
levelplot(correlations)
#plot(correlations, method="square")

print(correlations)
summary(correlations[upper.tri(correlations)])

#####################################################
# Pre-process  and feature selection--  before removing highly correlated vars
# run rf algorithm  and see predictors then decide which way is better
#####################################################
# confirm if any "NA" values in ds
any(is.na(iPhonefile))  

# remove obvious attributes
iPhonefile$id <- NULL

# change data types to integer
#iPhonefile$??? <- as.integer(iPhonefile$???)

# change data types to factor
#iPhonefile$??? <- factor(iPhonefile$???)

#delete vars that are highly correlated between themselves and also var that have very little correlation
# with the predicting var
correlations <- cor(iPhonefile)
highlyCorDescr <- findCorrelation(correlations, cutoff = .80) 
summary(highlyCorDescr) 
highlyCorDescr
 # [1] 47 52 43 44 38 39 15 17 40 14 45 22 18 13 30 32 24 28 11 16 21 46 51 34 55 31 26 54 57 58
#create a new dataset that excludes the list of highly correllated vars
iPhonefileNew <- iPhonefile[, -highlyCorDescr] 

str(iPhonefileNew)  #  13741 obs 29 var
#correlationsnew <- cor(iPhonefileNew[,c(1-30)])
correlationsnew <- cor(iPhonefileNew)
summary(correlationsnew[upper.tri(correlationsnew)])

histogram(iPhonefileNew$iphoneSentiment,data = NULL,plot=F)
corrplot(cor(iPhonefileNew), order ="hclust")
corrplot(correlationsnew, method = "circle")
levelplot(correlationsnew)
#plot(correlations, method="square")

print(correlationsnew)

str(iPhonefileNew)  #  13741 obs 29 var

# change predictor to factor containing 7 levels
disfixed7 <- discretize(iPhonefileNew$iphoneSentiment, "fixed", categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))
summary(disfixed7)
str(disfixed7)
#[-Inf, -50) [ -50, -10) [ -10,  -1) [  -1,   1) [   1,  10) [  10,  50) [  50,  Inf] 
#     36         124         314       10178         850        1925         314 

#insert vector into ds 
iPhonefileNew$iphoneSentiment <- disfixed7
#make level names more meaningful 
revalue(iPhonefileNew$iphoneSentiment, c("[-Inf, -50)"="verynegat", "[ -50, -10)"="negative ", "[ -10,  -1)"="swhtnegat", "[  -1,   1)"="neutral  ",
              "[   1,  10)"="smhtpostv", "[  10,  50)"="positive ", "[  50, Inf]"="verypostv"))

iPhonefileNew$iphoneSentiment <- mapvalues(iPhonefileNew$iphoneSentiment, from = c("[-Inf, -50)", "[ -50, -10)", "[ -10,  -1)", "[  -1,   1)", "[   1,  10)", "[  10,  50)", "[  50, Inf]"), 
          to = c("verynegat", "negative ", "swhtnegat", "neutral  ", "smhtpostv", "positive ", "verypostv"))

summary(iPhonefileNew)
  # iphoneSentiment
  # verynegat:   36
  # negative :  124
  # swhtnegat:  314
  # neutral  :10178
  # smhtpostv:  850
  # positive : 1925
  # verypostv:  314

str(iPhonefileNew)

#################################################################
# Train/test sets 
################################################################# 

set.seed(123) # set random seed - random selection can be reproduced

## create the training partition that is 70% of total obs 
inTraining <- createDataPartition(iPhonefileNew$iphoneSentiment, p=0.70, list=FALSE)
trainSet <- iPhonefileNew[inTraining,]   #  create training dataset 
testSet <- iPhonefileNew[-inTraining,]   #  create test/validate dataset

str(trainSet) # 50% 6870 obs of 29 var;  70% 9621 obs 29 var 
str(testSet)  # 50% 6871 obs of 29 var;  70% 4120 obs 29 var
#################################################################
# Train control
#################################################################

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
##fitControl <- rfeControl(method = "repeatedcv", number = 10, repeats = 10)

########################################
# Train and test model 

########################################
  #           Accuracy   Kappa  
  # ctree    0.8985855  0.7534092
  # rf       0.9052486  0.7703547  -- the best model
  # svm      0.8991687  0.7538190
  # c50      0.9037614  0.7657547  
  
# xgbTree  didn't run
  # knn    didnt' run for 70% train ds, with predicting var as factor with 7 levels
CTREEfit1 <- train(iphoneSentiment~., data = trainSet, method = "ctree", trControl = fitControl) 
CTREEfit1
   # mincriterion  Accuracy   Kappa  
   # 0.99          0.7899478  0.3903252   on  142 obs 29 var
   # 0.99          0.8999600  0.7557831   on 6871 obs 29 var  redicting var as int
   # 0.01          0.8991155  0.7545041  on 9621 obs and 28 predictors 7 clases '[-Inf, -50)', '[ -50, -10)'.....
   # 0.01          0.8985855  0.7534092  on 9621 obs and 28 predictors 7 clases as "verynegative", "negative "....
plot(CTREEfit1) 

xgbTreefit1 <- train(iphoneSentiment~., data = trainSet, method = "xgbTree", trControl = fitControl, allowParallel = TRUE)
xgbTreefit1

RFfit1 <- train(iphoneSentiment~., data = trainSet, method = "rf", trControl = fitControl, allowParallel = TRUE)
RFfit1
   # mtry  Accuracy   Kappa 
   # 28    0.8190724  0.5453002  on  142 obs 29 var
   # 15    0.9050780  0.7693372  on 6871 obs 29 var  predicting var as int
   # 15    0.9052716  0.7704195  on 9621 obs 28 predictors 7 clases '[-Inf, -50)', '[ -50, -10)'.....
   # 15    0.9052486  0.7703547  on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
plot(RFfit1)

KNNfit1 <- train(iphoneSentiment~., data = trainSet, method = "knn", trControl = fitControl)  #, linout = TRUE, metric = "Accuracy") 
KNNfit1 
   #  k  Accuracy   Kappa    
   #  5  0.8079721  0.4594152   on 142 obs 29 var as int
   #  didnt' run for 70% train ds, with predicting var as factor with 7 levels
plot(KNNfit1)

SVMfit1 <- train(iphoneSentiment~., data = trainSet, method = "svmLinear2", trControl = fitControl,allowParallel = TRUE) 
SVMfit1
  # cost  Accuracy   Kappa    
  # 0.25  0.8161807  0.5193008    on 142 obs 29 var   predicting as int
  # 0.25  0.9039153  0.7649496    on 6871 obs 29 var  predicting as int
  # 0.25  0.8991572  0.7537350    on 9621 obs 28 predictors 7 clases '[-Inf, -50)', '[ -50, -10)'.....
  # 0.25  0.8991687  0.7538190    on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
plot(SVMfit1)

C50fit1 <- train(iphoneSentiment~., data = trainSet, method = "C5.0", trControl = fitControl,allowParallel = TRUE) 
C50fit1
  # model  winnow  trials  Accuracy   Kappa
  # tree   FALSE    1      0.8155929  0.5620572  on 142 obs 29 var   predicting as vector
  # tree   FALSE   20      0.9034471  0.7643463  on 6871 obs 29 var  predicting as vector
  # tree   FALSE   20      0.9033257  0.7647155  on 9621 obs 28 predictors 7 clases '[-Inf, -50)', '[ -50, -10)'.....
  # tree   FALSE   20      0.9037614  0.7657547  on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....

plot(C50fit1)

summary(CTREEfit1)
summary(RFfit1)
summary(KNNfit1)
summary(SVMfit1)
summary(C50fit1)

# estimate variable importance
CTREEvarimp1 <- varImp(CTREEfit1, scale = FALSE)
RFvarimp1 <- varImp(RFfit1, scale = FALSE)  
KNNvarimp1 <- varImp(KNNfit1, scale = FALSE)  
SVMvarimp1 <- varImp(SVMfit1, scale = FALSE)
C50varimp1 <- varImp(C50fit1, scale = FALSE)

print(CTREEvarimp1,top = 28)
print(RFvarimp1,top = 28)   
print(KNNvarimp1,top = 28)  
print(SVMvarimp1,top = 28)
print(C50varimp1,top = 28)

#predictor variables
predictors(CTREEfit1)  
predictors(RFfit1)  
predictors(KNNfit1)  
predictors(SVMfit1)
predictors(C50fit1)

## save model object -- 
## saved in "C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Web Mining/"
saveRDS(CTREEfit1, "iPhone_ctree.rds")   
saveRDS(RFfit1, "iPhone_RF.rds")
saveRDS(KNNfit1, "iPhone_KNN.rds")
saveRDS(SVMfit1, "iPhone_SVM.rds")
saveRDS(C50fit1, "iPhone_C50.rds")

#############################################
## Predict using TRAIN ds  
#############################################
 # 	      Accuracy  Kappa
 # rf		0.932	0.741   the best results in Accuracy but c50 has better Kappa
 # svm		0.902	0.760
 # c50		0.917	0.797
 # ctree	0.907	0.775

# load and name model
#modeltrain <- readRDS("iPhone_ctree.rds")
#modeltrain <- readRDS("iPhone_RF.rds")
#modeltrain <- readRDS("iPhone_KNN.rds")
#modeltrain <- readRDS("iPhone_SVM.rds")
modeltrain <- readRDS("iPhone_C50.rds")

modelpredtrain <- predict(modeltrain, trainSet)  # predict on valid ds with trained model 
head(modelpredtrain,10) # output predicted values for each obs
tail(modelpredtrain,10) #
plot(modelpredtrain)

#plot predicted verses actual

comparison <- cbind(trainSet$iphoneSentiment, modelpredtrain)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtrain, trainSet$iphoneSentiment)

#####################################################################
## Predict using TEST ds  
#####################################################################
 #	     Accuracy	Kappa
 # rf		0.899	0.756
 # svm		0.894	0.742
 # c50		0.901	0.761
 # ctree	0.907	0.775  

# load and name model
#modeltrain <- readRDS("iPhone_ctree.rds")
#modeltest <- readRDS("iPhone_RF.rds")
#modeltest <- readRDS("iPhone_KNN.rds")
#modeltest <- readRDS("iPhone_SVM.rds")
modeltest <- readRDS("iPhone_C50.rds")

modelpredtest <- predict(modeltest, testSet)  # predict on valide ds with trained model 
head(modelpredtest,50) # output predicted values 
tail(modelpredtest,50)
plot(modelpredtest)

#head(modelpredtest)
#plot predicted verses actual

comparison <- cbind(testSet$iphoneSentiment, modelpredtest)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtest, testSet$iphoneSentiment)

################################################################
# resample 
################################################################
# Accuracy	Min.	1st Qu.	Median	Mean	3rd Qu.	Max.	NA's
# ctree		0.88	0.89	0.90	0.90	0.90	0.91	0
# rf		0.89	0.90	0.91	0.91	0.91	0.92	0
# svm		0.88	0.90	0.90	0.90	0.90	0.92	0
# C50		0.89	0.90	0.90	0.90	0.91	0.92	0
														
# Kappa		Min.	1st Qu.	Median	Mean	3rd Qu.	Max.	NA's
# ctree		0.711	0.740	0.755	0.753	0.766	0.793	0
# rf		0.732	0.758	0.771	0.770	0.782	0.804	0
# svm		0.710	0.744	0.753	0.754	0.764	0.803	0
# C50		0.722	0.757	0.766	0.766	0.774	0.816	0

resamps <- resamples(list(ctree = CTREEfit1, rf = RFfit1, knn = KNNfit1, svm = SVMfit1, C50 = C50fit1))
resamps <- resamples(list(ctree = CTREEfit1, rf = RFfit1, svm = SVMfit1, C50 = C50fit1))
summary(resamps)

diffs <- diff(resamps)
summary(diffs)

# ???????
# output <-  cbind(modelpred, iniPhonefile7v)
# #write.csv(output, file = "iniPhonefilectree.csv", row.names = FALSE)
# write.csv(output, file = "iniPhonefileRF.csv", row.names = FALSE)
# #write.csv(output, file = "iniPhonefileKNN.csv", row.names = FALSE)


