# Run data cleaning script
source('Data_Cleaning_MSc.R')


# Loading required packages
library(caret)
library(doParallel) # Parallel computations


# Parallelizing computations
cl <- makeCluster(15, outfile = '')
registerDoParallel(cl) # Change to acceptable number of cores based on your feasability
getDoParWorkers() 
stopCluster(cl) # Stop cluster computations



# Data Prep ----------------------------------------------------------------------

# Percent of data with at least one missing value
sum(complete.cases(data))/nrow(data) # 57% of data have no missing values

# Data with listwise deletion of missing values
data_noMiss <- na.omit(data) # Not desirable


# Holding 1/3 of the data as independent test set. Balanced partitioning. For original data
set.seed(337)
inTraining <- createDataPartition(data$Comp_30, p = 0.8, list = FALSE)
training <- data[inTraining, ]
testing <- data[-inTraining, ]



# Holding 1/3 of the data as independent test set. Balanced partitioning. For listwise deleted data
set.seed(337)
inTraining_noMiss <- createDataPartition(data_noMiss$Comp_30, p = 0.8, list = FALSE)
training_noMiss <- data_noMiss[inTraining_noMiss, ]
testing_noMiss <- data_noMiss[-inTraining_noMiss, ]


# Create reproducible folds
set.seed(337)
index <- createMultiFolds(y = data$Comp_30, times = 5)
index_noMiss <- createMultiFolds(y = training_noMiss$Comp_30, times = 5)


# 5 stat summary (ROC, Sens, Spec, Accuracy, Kappa) of model evaluation
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Outcome proportions for training and testing
table(testing$Comp_30)/sum(table(testing$Comp_30))
table(training$Comp_30)/sum(table(training$Comp_30))


# Prediction variables
predVars <- names(select(data, -c('Comp_30', 'Group')))


# Control objects -------------------------------------------------------------------
trCtrl <- trainControl(method = "repeatedcv",
                     repeats = 5,
                     summaryFunction = fiveStats,
                     classProbs = TRUE,
                     index = index_noMiss,
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

Basic_rfFull <- train(training_noMiss[, predVars],
                training_noMiss$Comp_30,
                method = "rf",
                metric = "ROC",
                tuneLength = 8,
                ntree = 1000,
                trControl = trCtrl)

confusionMatrix(predict(rfFull, testing_noMiss), testing_noMiss$Comp_30)

getModelInfo('nnet')
### Logistic Regression
Basic_logFull <- train(training_noMiss[, predVars],
                   training_noMiss$Comp_30,
                   method = 'glmStepAIC',
                   preProcess = c('center', 'scale'),
                   trace = 0,
                   trControl = trCtrl)
summary(Basic_log)




