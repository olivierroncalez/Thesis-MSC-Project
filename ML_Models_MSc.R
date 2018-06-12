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
index_original <- createMultiFolds(y = data$Comp_30, times = 5)
index_noMiss <- createMultiFolds(y = training_noMiss$Comp_30, times = 5)
index_facMiss <- createMultiFolds(y = training_facMiss$Comp_30, times = 5)


# 5 stat summary (ROC, Sens, Spec, Accuracy, Kappa) of model evaluation
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Prediction variables
predVars <- names(select(data, -c('Comp_30', 'Group')))






# Control objects -------------------------------------------------------------------
trCtrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     summaryFunction = fiveStats,
                     classProbs = TRUE,
                     index = index_facMiss,
                     allowParallel = TRUE,
                     verboseIter = TRUE)

# Control object for implementing recursive feature elimination
rfCtrl <- trainControl(method = "repeatedcv",
                         repeats = 5,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         index = index,
                         allowParallel = TRUE,
                         verboseIter = TRUE)

# Control object for implementing (needs "functions" argument)
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      index = index)





# ML Models (Basic Implementation) --------------------------------------------------


### Random Forest
set.seed(337)
rangerFull <- train(training_noMiss[, predVars],
                      training_noMiss$Comp_30,
                      trainControl = trCtrl,
                      preProc = c("center", "scale"),
                      num.trees = 20,
                      seed = 345, 
                      num.threads = 1,
                      importance = "permutation")

confusionMatrix(predict(rfFull, testing_noMiss), testing_noMiss$Comp_30)



### Logistic Regression
set.seed(337)
Basic_logFull <- train(training_noMiss[, predVars],
                      training_noMiss$Comp_30,
                      method = 'glm',
                      family = 'binomial',
                      preProcess = c('center', 'scale'),
                      trace = 0,
                      trControl = trCtrl)
summary(Basic_log)


### SVM
set.seed(337)
svmFull <- train(training_noMiss[, predVars],
                 training_noMiss$Comp_30,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 12,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)


### NB
nbFull <- train(training_noMiss[, predVars],
                training_noMiss$Comp_30,
                method = "nb",
                metric = "ROC",
                trControl = trCtrl)
nbFull
confusionMatrix(predict(nbFull, testing_noMiss), testing_noMiss$Comp_30)



### kNN
set.seed(337)
knnFull <- train(as.data.frame(training_noMiss),
                 training_noMiss$Comp_30,
                 method = "knn",
                 metric = "ROC",
                 tuneLength = 20,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)





# Test code -------------------------------------------------------------------------








