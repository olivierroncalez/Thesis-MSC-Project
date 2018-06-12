# Run data cleaning script
source('Data_Cleaning_MSc.R')


# Loading required packages
library(caret)
library(doParallel) # Parallel computations
library(mice) #for multiple imputation and examining missing data patterns.


# Parallelizing computations 
cl <- makeCluster(7, outfile = '')
registerDoParallel(cl) # Change to acceptable number of cores based on your feasability
getDoParWorkers()
stopCluster(cl) # Stop cluster computations
registerDoSEQ() # Unregister doParallel


# Converting back to dataframe from tibble (caret is not tibble friendly)
data <- as.data.frame(data)
data_facMiss <- as.data.frame(data_facMiss)
data_noMiss <- as.data.frame(data_noMiss)

# Data Prep ----------------------------------------------------------------------


# Holding .2 of the data as independent test set. Balanced partitioning. For original data
set.seed(337)
inTraining_original <- createDataPartition(data$Comp_30, p = 0.8, list = FALSE)
training_original <- data[inTraining_original, ]
testing_original <- data[-inTraining_original, ]


# Holding .2 of the data as independent test set. Balanced partitioning. For listwise deleted data
set.seed(337)
inTraining_noMiss <- createDataPartition(data_noMiss$Comp_30, p = 0.8, list = FALSE)
training_noMiss <- data_noMiss[inTraining_noMiss, ]
testing_noMiss <- data_noMiss[-inTraining_noMiss, ]


# Holding .2 of the data as independent test set. Balanced partitioning. For missing factor data
set.seed(337)
inTraining_facMiss <- createDataPartition(data_facMiss$Comp_30, p = 0.8, list = FALSE)
training_facMiss <- data_facMiss[inTraining_facMiss, ]
testing_facMiss <- data_facMiss[-inTraining_facMiss, ]


# Create reproducible folds
set.seed(337)
index_original <- createMultiFolds(y = data$Comp_30, times = 3)
index_noMiss <- createMultiFolds(y = training_noMiss$Comp_30, times = 3)
index_facMiss <- createMultiFolds(y = training_facMiss$Comp_30, times = 3)


# 5 stat summary (ROC, Sens, Spec, Accuracy, Kappa) of model evaluation
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Prediction variables
predVars <- names(select(data, -c('Comp_30', 'Group')))





# Control objects -------------------------------------------------------------------


### Original Control


# Cross-fold validation control
trCtrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     summaryFunction = fiveStats,
                     classProbs = TRUE,
                     index = dummy_index_full, # IMPORTANT! 
                     allowParallel = TRUE,
                     verboseIter = TRUE,
                     savePredictions = TRUE)





### Recursive Feature Elimination


# Control object for implementing recursive feature elimination
rfCtrl <- rfeControl(method = "repeatedcv",
                         repeats = 2,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         index = index_noMiss, # IMPORTANT! 
                         allowParallel = TRUE,
                         verboseIter = TRUE)





### Selection by Filter


# Control object for implementing (needs "functions" argument)
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      index = index_noMiss # IMPORTANT! 
                      )





# ML Models (Basic Implementation) --------------------------------------------------


### Logistic Regression


set.seed(337)
logisticFull <- train(data_dummied[, -ncol(data_dummied)],
                      data_dummied$Comp_30,
                      method = 'glm',
                      family = 'binomial',
                      preProcess = c('center', 'scale', 'medianImpute'),
                      trace = 0,
                      trControl = trCtrl)
summary(logisticFull)
# confusionMatrix(predict(logisticFull, testing_dummied), testing_dummied$Comp_30)






### Random Forest


set.seed(337)
rfFull <- train(data_dummied[, -ncol(data_dummied)],
                data_dummied$Comp_30,
                      method = "rf",
                      metric = "ROC",
                      tuneLength = 4,
                      ntree = 1000,
                preProcess = 'knnImpute',
                      trControl = trCtrl)

confusionMatrix(predict(rfFull, testing_dummied), testing_dummied$Comp_30)






### SVM


set.seed(337)
svmFull <- train(data_dummied[, -ncol(data_dummied)],
                 data_dummied$Comp_30,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 6,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)
confusionMatrix(predict(svmFull, testing_dummied, type = 'prob'), testing_dummied$Comp_30) # Not working



### NB


nbFull <- train(data_dummied[, -ncol(data_dummied)],
                data_dummied$Comp_30,
                method = "nb",
                metric = "Kappa",
                preProcess = 'medianImpute',
                trControl = trCtrl)
nbFull
confusionMatrix(predict(nbFull, testing_dummied), testing_dummied$Comp_30)





### kNN


set.seed(337)
knnFull <- train(data_dummied[ , -ncol(data_dummied)],
                 data_dummied$Comp_30,
                 method = "kknn",
                 metric = "Kappa",
                 tuneLength = 10,
                 preProc = c("center", "scale", 'medianImpute'),
                 trControl = trCtrl)

confusionMatrix(predict(knnFull, testing_dummied), testing_dummied$Comp_30)
plot(knnFull)




# Test code -------------------------------------------------------------------------


### Dummy variables for no missing data


data_dummied <- data_noMiss
data_dummied %<>% select(-Group)
dummies <- dummyVars(~., select(data_dummied, -Comp_30), fullRank = TRUE)
data_dummied <- as.data.frame(predict(dummies, newdata = data_dummied)) %>% 
     bind_cols(., data_dummied[ncol(data_dummied)])



nearZeroVar(data_dummied, saveMetrics = TRUE) %>% rownames_to_column(var = 'Variable') %>% 
     filter(nzv == TRUE)
nzv <- nearZeroVar(data_dummied, saveMetrics = FALSE)
data_dummied %<>% select(-nzv)


indexpartition <- createDataPartition(data_dummied$Comp_30, p = .8, list = FALSE)
training_dummied <- data_dummied[indexpartition, ]
testing_dummied <- data_dummied[-indexpartition, ]






### Dummy variables for encoded missing data in factors


data_dummied <- data_facMiss
data_dummied %<>% select(-Group)
dummies <- dummyVars(~., select(data_dummied, -Comp_30), fullRank = TRUE)
data_dummied <- as.data.frame(predict(dummies, newdata = data_dummied)) %>% 
     bind_cols(., data_dummied[ncol(data_dummied)])


nearZeroVar(data_dummied, saveMetrics = TRUE) %>% rownames_to_column(var = 'Variable') %>% 
     filter(nzv == TRUE)
nzv <- nearZeroVar(data_dummied, saveMetrics = FALSE)
data_dummied %<>% select(-nzv)


indexpartition <- createDataPartition(data_dummied$Comp_30, p = .8, list = FALSE)
training_dummied <- data_dummied[indexpartition, ]
testing_dummied <- data_dummied[-indexpartition, ]
dummy_index <- createMultiFolds(training_dummied$Comp_30, times = 3)
dummy_index_full <- createMultiFolds(data_dummied$Comp_30, times = 3)


nearZeroVar(training_dummied, saveMetrics = TRUE) %>% rownames_to_column(var = 'Variable') %>% 
     filter(nzv == TRUE | zeroVar == TRUE)





# Imputation Methods ----------------------------------------------------------------


init <- mice(data, maxit = 0)
meth <- init$method
predM <- init$predictorMatrix

predM[, c('Group', 'Comp_30')] <- 0

set.seed(103)
imputed <- mice(data, method=meth, predictorMatrix=predM, m=5)




data %>% sapply(function(x) is.numeric(x))
table(data$RESP)
mice(data,m=5,maxit=50,meth='pmm',seed=500)

imputed <- hot.deck(data = data, m = 5, cutoff = 10, impContinuous = 'mice')

sum(is.na(imputed$data[[1]]))


