#
# Project name: Overal Sentiment Toward Galaxy (data collected from the web using AWS)
# File name:    Galaxyfile (70% - train and 30% -test ds) 
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
#install.packages("rattle", repos="http://rattle.togaware.com", type="source") 
install.packages("rattle")  #for graphics
install.packages("arules")
install.packages("xgboost")
install.packages("kknn")

require(doParallel)
require(plyr)  
require(party)
require(caret)  
require(corrplot)
require(randomForest)
require(C50)
require(rattle)
require(lattice)
require(arules)
require(xgboost)
require(kknn)

#the following lines will create a local 4-node snow cluster for faster execution
workers=makeCluster(4,type="SOCK")
registerDoParallel(workers)
foreach(i=1:4) %dopar% Sys.getpid()

############################################################
# Import data
############################################################

#read input data file
Galaxyfile <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Web Mining/Output Files/Wet100Run/GalaxyLargeMatrix.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)

# view data structure                
str(Galaxyfile)  # 13,741 obs of 60 vars 

# save a copy of input file in working directory before any changes made
write.csv(Galaxyfile, "Galaxyfile_sentimet.csv")

#######################################
# Evaluate data
####################################### 

head(Galaxyfile,5)
tail(Galaxyfile,5)
summary(Galaxyfile)

options(max.print=1000000) # print default is only 20 vars

histogram(Galaxyfile$galaxySentiment, data = NULL,plot=F)
 
# can use either of the two lines below, however when specifying columns the predicting var should not be included
# correlations <- cor(Galaxyfile[,c(1-59)])
correlations <- cor(Galaxyfile)
corrplot(correlations, method = "circle")
corrplot(correlations, order ="hclust")
levelplot(correlations)
#plot(correlations, method="square")

print(correlations)
summary(correlations[upper.tri(correlations)])

#####################################################
# Pre-process 
#####################################################
# confirm if any "NA" values in ds
any(is.na(Galaxyfile))  

# remove obvious attributes
Galaxyfile$id <- NULL

# change data types to integer
#Galaxyfile$??? <- as.integer(Galaxyfile$???)

# change data types to factor
#Galaxyfile$??? <- factor(Galaxyfile$???)

#####################################################
# Feature selection--  before removing highly correlated vars
# run rf algorithm  and see predictors then decide which way is better
#####################################################
set.seed(123) # set random seed - random selection can be reproduced

## create partition that is 15 of total obs just to run rf to determine predictors 
featureselection <- createDataPartition(Galaxyfile$galaxySentiment, p=0.40, list=FALSE)
fselectionSet <- Galaxyfile[featureselection,]   #  create feature selection dataset

#fitControl <- rfeControl(functions = lmFuncs, method = "cv", number = 10) # returned 59 predictors
fitControl <- rfeControl(functions = rfFuncs, method = "cv", number = 10)
results <- rfe(fselectionSet, fselectionSet$galaxySentiment, rfeControl = fitControl)

predictors(results) # retured the following 16 predictors when running only 15% of data, 
                    # but when running 40% -- 59 predictors were selected 
  # "galaxySentiment" "googleperneg"    "googleperpos"    "samsungdispos"   "samsungcampos"   "googleperunc"   
# "iphonedispos"    "samsungdisneg"   "samsungperpos"   "samsungperneg"   "samsungcamunc"   "samsungcamneg"  
# "iphonecampos"    "htccamneg"       "ios"             "samsungdisunc"  
print(results)
  # Variables  RMSE  Rsquared RMSESD RsquaredSD 
  #    16      4.757  0.9335  4.068  0.05859  when running 15% of data
  #    59      4.069  0.9495  3.954  0.04900  when running 40% of data
#find vars that are highly correlated between themselves 
correlations <- cor(Galaxyfile)
highlyCorDescr <- findCorrelation(correlations, cutoff = .80) 
summary(highlyCorDescr) 
highlyCorDescr
 # [1] 47 52 43 44 38 39 15 17 40 14 45 22 18 13 30 32 24 28 11 16 21 46 51 34 55 31 26 54 57 58 -- for iPhone
 # [1] 47 44 52 43 39 38 15 17 19 40 45 22 18 13 30 32 24 28 11 16 21 46 51 34 55 31 26 54 58 57 -- for Galaxy

#create a new dataset that excludes the list of highly correllated vars
GalaxyfileNew <- Galaxyfile[, -highlyCorDescr] 

str(GalaxyfileNew)  #  13741 obs 30 var
#correlationsnew <- cor(GalaxyfileNew[,c(1-30)])
correlationsnew <- cor(GalaxyfileNew)
summary(correlationsnew[upper.tri(correlationsnew)])

histogram(GalaxyfileNew$galaxySentiment,data = NULL,plot=F)
corrplot(cor(GalaxyfileNew), order ="hclust")
corrplot(correlationsnew, method = "circle")
levelplot(correlationsnew)
#plot(correlations, method="square")

print(correlationsnew)

str(GalaxyfileNew)  #  13741 obs 30 var

# change predictor to factor containing 7 levels
disfixed7 <- discretize(GalaxyfileNew$galaxySentiment, "fixed", categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))
summary(disfixed7)
 #[-Inf, -50) [ -50, -10) [ -10,  -1) [  -1,   1) [   1,  10) [  10,  50) [  50,  Inf] ## this was for iPhone
 #     36         124         314       10178         850        1925         314    
############################################################################################################### 
  ##[-Inf, -50) [ -50, -10) [ -10,  -1) [  -1,   1) [   1,  10) [  10,  50) [  50, Inf]   ## this is for Galaxy
   ##      8          70          75       12408         361         649         170

#insert vector into ds 
GalaxyfileNew$galaxySentiment <- disfixed7
#make level names more meaningful 
revalue(GalaxyfileNew$galaxySentiment, c("[-Inf, -50)"="verynegat", "[ -50, -10)"="negative ", "[ -10,  -1)"="swhtnegat", "[  -1,   1)"="neutral  ",
              "[   1,  10)"="smhtpostv", "[  10,  50)"="positive ", "[  50, Inf]"="verypostv"))

GalaxyfileNew$galaxySentiment <- mapvalues(GalaxyfileNew$galaxySentiment, from = c("[-Inf, -50)", "[ -50, -10)", "[ -10,  -1)", "[  -1,   1)", "[   1,  10)", "[  10,  50)", "[  50, Inf]"), 
          to = c("verynegat", "negative ", "swhtnegat", "neutral  ", "smhtpostv", "positive ", "verypostv"))

summary(GalaxyfileNew)
  # IphoneSentiment  GalaxSentiment
  # verynegat:   36       8
  # negative :  124      70
  # swhtnegat:  314      75
  # neutral  :10178   12408
  # smhtpostv:  850     361
  # positive : 1925     649
  # verypostv:  314     170

str(GalaxyfileNew)

#################################################################
# Train/test sets 
################################################################# 

set.seed(123) # set random seed - random selection can be reproduced

## create partition 70% for train ds and 30% for test ds
inTraining <- createDataPartition(GalaxyfileNew$galaxySentiment, p=0.70, list=FALSE)
trainSet <- GalaxyfileNew[inTraining,]   #  create training dataset 
testSet <- GalaxyfileNew[-inTraining,]   #  create test/validate dataset

str(trainSet) #  70% 9621 obs 29 var 
str(testSet)  #  30% 4120 obs 29 var
#################################################################
# Train control
#################################################################

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
##fitControl <- rfeControl(method = "repeatedcv", number = 10, repeats = 10)

########################################
# Train Model
########################################
  #           Accuracy   Kappa  
  # ctree    0.9545898  0.7206293
  # xgbTree  0.9662625  0.8060307
  # rf       0.9672910  0.8127568
  # kknn     0.9599111  0.7680013
  # svm      0.9528122  0.7116858
  # c50      0.9650361  0.7976591
  
CTREEfit1 <- train(galaxySentiment~., data = trainSet, method = "ctree", trControl = fitControl) 
CTREEfit1
   # mincriterion  Accuracy   Kappa  
   # 0.01          0.9545898 0.7206293  on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
plot(CTREEfit1) 

xgbTreefit1 <- train(galaxySentiment~., data = trainSet, method = "xgbTree", trControl = fitControl, allowParallel = TRUE)
    
    #  eta  max_depth  colsample_bytree  subsample  nrounds  Accuracy   Kappa 
    #  0.4  3          0.8               0.75       150      0.9662625  0.8060307
xgbTreefit1

RFfit1 <- train(galaxySentiment~., data = trainSet, method = "rf", trControl = fitControl, allowParallel = TRUE)
RFfit1
   # mtry  Accuracy   Kappa 
   # 15    0.9672910  0.8127568  on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
plot(RFfit1)

KNNfit1 <- train(galaxySentiment~., data = trainSet, method = "kknn", trControl = fitControl)  #, linout = TRUE, metric = "Accuracy") 
KNNfit1 
   # knn didn't work, ran kknn instead
   # kmax  Accuracy   Kappa    
   # 0.9599111  0.7680013  # distance = 2 and kernel = optimal
plot(KNNfit1)

SVMfit1 <- train(galaxySentiment~., data = trainSet, method = "svmLinear2", trControl = fitControl,allowParallel = TRUE) 
SVMfit1
  # cost  Accuracy   Kappa    
  # 0.25  0.9528122  0.7116858    on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
plot(SVMfit1)

C50fit1 <- train(galaxySentiment~., data = trainSet, method = "C5.0", trControl = fitControl,allowParallel = TRUE) 
C50fit1
  # model  winnow  trials  Accuracy   Kappa 
  # tree    TRUE   20      0.9650361  0.7976591  on 9621 obs 28 predictors 7 clases as "verynegative", "negative "....
  # all model winnow and trials results were the same

plot(C50fit1)

summary(CTREEfit1)
summary(xgbTreefit1)
summary(RFfit1)
summary(KNNfit1)
summary(SVMfit1)
summary(C50fit1)
#(a)   (b)   (c)   (d)   (e)   (f)   (g)    <-classified as
#----  ----  ----  ----  ----  ----  ----
#  5     1                                  (a): class verynegat
# 43     1     1     3     1          (b): class negative
# 34     3    12     3     1    (c): class swhtnegat
# 2  8673    11                (d): class neutral
# 6     2    16   225     4          (e): class smhtpostv
# 1     1    77    37   338     1    (f): class positive
# 1     2     1    15   100    (g): class verypostv

#Attribute usage:
#100.00%	samsungcampos
#100.00%	samsungcamneg
#100.00%	iphonedisunc
#100.00%	sonydisunc
#100.00%	samsungperunc
#100.00%	googleperneg
# 99.99%	samsungdisneg
# 99.98%	htcdispos
# 99.92%	iphonedispos
# 99.72%	htcperpos
# 99.43%	iphoneperunc
# 99.42%	samsunggalaxy
# 99.40%	googleandroid
# 98.72%	iphone
# 98.46%	iosperpos
# 97.99%	htcphone
# 97.63%	iphonecampos
# 96.89%	sonyxperia
# 96.56%	ios

## estimate variable importance 
CTREEvarimp1 <- varImp(CTREEfit1, scale = FALSE)
xgbvarimp1 <- varImp(xgbTreefit1, scale = FALSE)
RFvarimp1 <- varImp(RFfit1, scale = FALSE)  
KNNvarimp1 <- varImp(KNNfit1, scale = FALSE)  
SVMvarimp1 <- varImp(SVMfit1, scale = FALSE)
C50varimp1 <- varImp(C50fit1, scale = FALSE)

print(CTREEvarimp1,top = 28)
print(xgbvarimp1,top = 28)
print(RFvarimp1,top = 28)   
print(KNNvarimp1,top = 28)  
print(SVMvarimp1,top = 28)
print(C50varimp1,top = 28)

#predictor variables
predictors(CTREEfit1)  
predictors(xgbTreefit1) 
predictors(RFfit1)  
predictors(KNNfit1)  
predictors(SVMfit1)
predictors(C50fit1)

## save model object -- 
## saved in "C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Web Mining/"

saveRDS(xgbTreefit1, "Galaxy_xgbtree")
saveRDS(CTREEfit1, "Galaxy_ctree.rds")   
saveRDS(RFfit1, "Galaxy_RF.rds")
saveRDS(KNNfit1, "Galaxy_KNN.rds")
saveRDS(SVMfit1, "Galaxy_SVM.rds")
saveRDS(C50fit1, "Galaxy_C50.rds")

#############################################
## Predict using TRAIN ds  
#############################################
  #     Accuracy	Kappa
  # xgb	  0.983	0.901
  # ctree	0.956	0.730
  # rf	  0.985	0.914  the best 
  # knn	  0.976	0.858
  # svm	  0.958	0.743
  # c50	  0.979	0.879


# load and name model
#modeltrain <- readRDS("Galaxy_xgbtree")
#modeltrain <- readRDS("Galaxy_ctree.rds")
#modeltrain <- readRDS("Galaxy_RF.rds")
#modeltrain <- readRDS("Galaxy_KNN.rds")
#modeltrain <- readRDS("Galaxy_SVM.rds")
modeltrain <- readRDS("Galaxy_C50.rds")

modelpredtrain <- predict(modeltrain, trainSet)  # predict on valid ds with trained model 
head(modelpredtrain,10) # output predicted values for each obs
tail(modelpredtrain,10) #
plot(modelpredtrain)

#plot predicted verses actual

comparison <- cbind(trainSet$galaxySentiment, modelpredtrain)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtrain, trainSet$galaxySentiment)

#####################################################################
## Predict using TEST ds  
#####################################################################
  #     Accuracy	Kappa
  # xgb	  0.963	  0.785
  # ctree	0.955  	0.724
  # rf	  0.955	  0.797  the best because Kappa is much higher
  # knn	  0.960	  0.762
  # svm	  0.954	  0.717
  # c50	  0.964	  0.788

# load and name model
#modeltest <- readRDS("Galaxy_xgbtree")
#modeltest <- readRDS("Galaxy_ctree.rds")
modeltest <- readRDS("Galaxy_RF.rds")
#modeltest <- readRDS("Galaxy_KNN.rds")
#modeltest <- readRDS("Galaxy_SVM.rds")
#modeltest <- readRDS("Galaxy_C50.rds")

modelpredtest <- predict(modeltest, testSet)  # predict and validate ds with trained model 
head(modelpredtest,50) # output predicted values 
tail(modelpredtest,50)
plot(modelpredtest)

#head(modelpredtest)
#plot predicted verses actual

comparison <- cbind(testSet$galaxySentiment, modelpredtest)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtest, testSet$galaxySentiment)

################################################################
# resample 
###############################################################
  # Accuracy							
  #         Min.	1st Qu.	Median	Mean	  3rd Qu.	Max.	  NA's
  # xgb	  0.953	  0.964	  0.966	  0.966	  0.970	  0.979	  0
  # ctree	0.943	  0.951	  0.954	  0.955	  0.959 	0.965 	0
  # rf	  0.955	  0.964	  0.967	  0.967	  0.971	  0.981 	0
  # svm	  0.943	  0.950	  0.953	  0.953 	0.956 	0.965 	0
  # C50	  0.952	  0.961	  0.965	  0.965 	0.969	  0.978	  0

  # Kappa							
  #         Min.	1st Qu.	Median	Mean  	3rd Qu.	Max.	NA's
  # xgb	  0.719	  0.790 	0.805	  0.806	  0.824	  0.878 	0
  # ctree	0.645	  0.695 	0.720	  0.721 	0.750	  0.792	  0
  # rf	  0.744 	0.791 	0.806 	0.813 	0.833 	0.895 	0
  # svm	  0.630 	0.692 	0.715 	0.712 	0.731 	0.788 	0
  # C50	  0.719 	0.775	  0.797 	0.798 	0.824 	0.874 	0


resamps <- resamples(list(xgb = xgbTreefit1, ctree = CTREEfit1,  rf = RFfit1, kknn = KNNfit1, svm = SVMfit1, C50 = C50fit1))
summary(resamps)

diffs <- diff(resamps)
summary(diffs)

 output <-  cbind(modelpredtest, testSet)
# write.csv(output, file = "inGalaxyoutputctree.csv", row.names = FALSE)
# write.csv(output, file = "inGalaxyoutputxgb.csv", row.names = FALSE)
# write.csv(output, file = "inGalaxyoutputsvm.csv", row.names = FALSE)
  write.csv(output, file = "inGalaxyoutputRF.csv", row.names = FALSE)
# write.csv(output, file = "inGalaxyoutputKNN.csv", row.names = FALSE)
# write.csv(output, file = "inGalaxyoutputc50.csv", row.names = FALSE)

