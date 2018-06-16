# Run data cleaning script
source('Data_Cleaning_MSc.R')


# Loading required packages
library(caret)
library(caretEnsemble)
library(doParallel) # Parallel computations


# Parallelizing computations 
cl <- makeCluster(15, outfile = '')
registerDoParallel(cl) # Change to acceptable number of cores based on your feasability
getDoParWorkers()
stopCluster(cl) # Stop cluster computations
registerDoSEQ() # Unregister doParallel


# Converting back to dataframe from tibble (caret is not tibble friendly)
data <- as.data.frame(data)
data_facMiss <- as.data.frame(data_facMiss)
data_noMiss <- as.data.frame(data_noMiss)
data_facMiss_dummied <- as.data.frame(data_facMiss_dummied)
data_facMiss_dummied_cat <- as.data.frame(data_facMiss_dummied_cat)







# Data Prep ----------------------------------------------------------------------


### Training/Testing Split


# Holding .2 of the data as independent test set. Balanced partitioning. For dummied missing factor data
set.seed(337)
idx <- createDataPartition(data_facMiss_dummied$Comp_30, p = .8, list = FALSE) # Creating index to partition data
training_dummied_fac_Miss <- data_facMiss_dummied[idx, ] # Training
testing_dummied_fac_Miss <- data_facMiss_dummied[-idx, ] # Testing
training_dummied_fac_Miss_cat <- data_facMiss_dummied_cat[idx, ] # Training (same but recoded as factors)
testing_dummied_fac_Miss_cat <- data_facMiss_dummied_cat[-idx, ] # Testing (same data but recoded as factors)
rm('idx')


# Holding .2 of the data as independent test set. Balanced partitioning. For listwise deleted data
set.seed(337)
inTraining_noMiss <- createDataPartition(data_noMiss$Comp_30, p = 0.8, list = FALSE)
training_noMiss <- data_noMiss[inTraining_noMiss, ]
testing_noMiss <- data_noMiss[-inTraining_noMiss, ]
rm('inTraining_noMiss') # Remove index


# Holding .2 of the data as independent test set. Balanced partitioning. For missing factor data
set.seed(337)
inTraining_facMiss <- createDataPartition(data_facMiss$Comp_30, p = 0.8, list = FALSE)
training_facMiss <- data_facMiss[inTraining_facMiss, ]
testing_facMiss <- data_facMiss[-inTraining_facMiss, ]
rm('inTraining_facMiss') # Remove index




### Cross-Fold Validation index generation

# Create reproducible folds (10 fold validation 3 times) - ensures same folds used to train different models. Mainly used as validation set for models
# requiring parameter tuning, but can also provide some interesting insight.
set.seed(337)
dummy_index <- createMultiFolds(training_dummied_fac_Miss$Comp_30, times = 3) # Dummied factor missing data
set.seed(337)
index_noMiss <- createMultiFolds(y = training_noMiss$Comp_30, times = 3) # No missing data
set.seed(337)
index_facMiss <- createMultiFolds(y = training_facMiss$Comp_30, times = 3) # Factor missing data





### Additional objects


# 5 stat summary (ROC, Sens, Spec, Accuracy, Kappa) of model evaluation
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Prediction variables
predVars <- names(select(training_dummied_fac_Miss, -Comp_30)) # Predictor names






# Control objects -------------------------------------------------------------------


####################################
### Original Control
####################################


# Cross-fold validation control (computational nuances)
trCtrl <- trainControl(method = "repeatedcv",
                     repeats = 3,
                     summaryFunction = fiveStats,
                     classProbs = TRUE,
                     index = dummy_index, # IMPORTANT! 
                     allowParallel = TRUE,
                     verboseIter = TRUE,
                     savePredictions = TRUE)





####################################
### Recursive Feature Elimination
####################################


# Control object for implementing recursive feature elimination
rfCtrl <- rfeControl(method = "repeatedcv",
                         repeats = 3,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         index = dummy_index, # IMPORTANT! 
                         allowParallel = TRUE,
                         verboseIter = TRUE)





####################################
### Selection by Filter
####################################


# Control object for implementing (needs "functions" argument)
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      index = index_noMiss # IMPORTANT! 
                      )






# ML Models (Basic Implementation) --------------------------------------------------


####################################
### Logistic Regression
####################################


### Standard Logistic Regression

set.seed(337)
logisticFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                      training_dummied_fac_Miss$Comp_30,
                      method = 'glm',
                      family = 'binomial',
                      preProcess = c('center', 'scale'),
                      trace = 0, # No verbose printout
                      trControl = trCtrl)
summary(logisticFull) # GLM model info
logisticFull # Caret model info


logisticFull_pred <- predict(logisticFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(logisticFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix




### AIC Log


set.seed(337)
LogAICFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                      training_dummied_fac_Miss$Comp_30,
                      method = 'glmStepAIC',
                      family = 'binomial',
                      preProcess = c('center', 'scale'),
                      trace = 0, # No verbose printout
                      trControl = trCtrl)
summary(LogAICFull) # GLM model info
LogAICFull # Caret model info


logisticFull_pred <- predict(logisticFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(logisticFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix






####################################
### Random Forest
####################################


# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


# Grid of tuning parameters to try
rf_grid <- expand.grid(mtry = c(1:10))

set.seed(337)
rfFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                      method = "rf",
                      metric = "ROC",
                      tuneGrid = rf_grid,
                      ntree = 1000,
                      trControl = trCtrl)
rfFull # Model info


rfFull_pred <- predict(rfFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(rfFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix





####################################
### SVM
####################################


set.seed(337)
svmFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                 training_dummied_fac_Miss$Comp_30,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 6,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)
svmFull # Model info





####################################
### NB
####################################


# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


set.seed(337)
nbFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                method = "nb",
                metric = "ROC",
                preProcess = 'medianImpute',
                trControl = trCtrl)
nbFull # Model info


nbFull_pred <- predict(nbFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(nbFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix





####################################
### kNN
####################################


set.seed(337)
knnFull <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                 training_dummied_fac_Miss$Comp_30,
                 method = "kknn",
                 metric = "ROC",
                 tuneLength = 10,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)
knnFull # Model info


knnFull_pred <- predict(knnFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(knnFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix





####################################
### Neural Network
####################################


set.seed(337)
nnetFull <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  method = 'nnet',
                  tuneLength = 4,
                  trace = FALSE, lineout = TRUE,
                  trControl = trCtrl)
nnetFull # Model info


nnetFull_pred <- predict(nnetFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(nnetFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix





# Model Comparisons  ----------------------------------------------------------------


model_list <- list('Logistic Reg' = logisticFull,
                   'Random Forest' = rfFull,
                   'SVM' = svmFull,
                   'N. Bayes' = nbFull,
                   'kNN' = knnFull,
                   'Neural Net' = nnetFull)

resample_ <- resamples(model_list) # Currently not working...

summary(resample_)



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











# Imputation Methods ----------------------------------------------------------------


### Hot Deck Imputation

# Generating imputed datasets
imputed <- hot.deck(data = data, m = 5, cutoff = 10, impContinuous = 'mice')
sum(is.na(imputed$data[[1]]))
nearZeroVar(imputed$data[[1]], saveMetrics = TRUE)

