# Run data cleaning script
source('Data_Cleaning_MSc.R')


# Loading required packages
library(caret)
library(caretEnsemble)
library(doParallel) # Parallel computations
library(xtable) # LaTeX table output
library(pROC) # ROC curves


# Parallelizing computations 
cl <- makeCluster(15, outfile = '')
registerDoParallel(cl) # Change to acceptable number of cores based on your feasability
getDoParWorkers()
stopCluster(cl) # Stop cluster computations
registerDoSEQ() # Unregister doParallel


# Converting back to dataframe from tibble (caret is not tibble friendly)
# data <- as.data.frame(data)
data_facMiss <- as.data.frame(data_facMiss) # Grouped predictors
# data_noMiss <- as.data.frame(data_noMiss)
data_facMiss_dummied <- as.data.frame(data_facMiss_dummied)
data_facMiss_dummied_cat <- as.data.frame(data_facMiss_dummied_cat)


# Glance at Data
# head(data, 15) # Original data
head(data_facMiss, 15) # USED: Grouped predictors (categorical variables)
# head(data_noMiss, 15) # Listwise imputation
head(data_facMiss_dummied, 15) # USED: Independent predictors (dummy variables)
head(data_facMiss_dummied_cat, 15) # Categories recategorized as factors (used for NB)


# Remove unused data
rm('data', 'data_noMiss')






# Data Prep ----------------------------------------------------------------------


########################################################################
### Training/Testing Split


# Holding .2 of the data as independent test set. Balanced partitioning. For dummied missing factor data
set.seed(337)
idx <- createDataPartition(data_facMiss_dummied$Comp_30, p = .8, list = FALSE) # Creating index to partition data

training_dummied_fac_Miss <- data_facMiss_dummied[idx, ] # Training
testing_dummied_fac_Miss <- data_facMiss_dummied[-idx, ] # Testing

training_dummied_fac_Miss_cat <- data_facMiss_dummied_cat[idx, ] # Training (same but recoded as factors)
testing_dummied_fac_Miss_cat <- data_facMiss_dummied_cat[-idx, ] # Testing (same data but recoded as factors)
rm('idx')


# Holding .2 of the data as independent test set. Balanced partitioning. For missing factor data
set.seed(337)
idx <- createDataPartition(data_facMiss$Comp_30, p = 0.8, list = FALSE)
training_facMiss <- data_facMiss[idx, ]
testing_facMiss <- data_facMiss[-idx, ]
rm('idx') # Remove index


# # Holding .2 of the data as independent test set. Balanced partitioning. For listwise deleted data
# set.seed(337)
# inTraining_noMiss <- createDataPartition(data_noMiss$Comp_30, p = 0.8, list = FALSE)
# training_noMiss <- data_noMiss[inTraining_noMiss, ]
# testing_noMiss <- data_noMiss[-inTraining_noMiss, ]
# rm('inTraining_noMiss') # Remove index




########################################################################
### Cross-Fold Validation index generation

# Create reproducible folds (10 fold validation 3 times) - ensures same folds used to train different models. Mainly used as validation set for models
# requiring parameter tuning, but can also provide some interesting insight. Given that dummied factor missing data, non-dummied data, 
# and dummied non factor data have the same identical rows, only one index is needed.
set.seed(337)
dummy_index <- createMultiFolds(training_dummied_fac_Miss$Comp_30, times = 3) # Dummied factor missing data
# set.seed(337)
# index_noMiss <- createMultiFolds(y = training_noMiss$Comp_30, times = 3) # No missing data




########################################################################
### Additional objects


# 5 stat summary (ROC, Sens, Spec, Accuracy, Kappa) of model evaluation
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Prediction variable names
predVars_dummy <- names(select(training_dummied_fac_Miss, -Comp_30)) # Predictor names for dummied factor data
predVars <- names(select(training_facMiss, -Comp_30)) # Predictor namesfor grouped categorical vars







# Control objects -------------------------------------------------------------------


########################################################################
### 
###                      Original Control
### 
########################################################################


# Cross-fold validation control (computational nuances)
trCtrl <- trainControl(method = "repeatedcv", # Defaults to 10
                       repeats = 1,
                       summaryFunction = fiveStats,
                       index = dummy_index, # IMPORTANT! Uses the same index for all models
                       classProbs = TRUE,
                       allowParallel = TRUE,
                       verboseIter = TRUE,
                       savePredictions = TRUE,
                       returnResamp = 'final')






# Models ----------------------------------------------------------------------------



########################################################################
### 
###                      Logistic Regression
### 
########################################################################



########################################################################
### AIC Log


set.seed(337)
LogAICFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                    training_dummied_fac_Miss$Comp_30,
                    method = 'glmStepAIC',
                    family = 'binomial',
                    preProcess = c('center', 'scale'),
                    metric = 'ROC',
                    trace = 0, # No verbose printout
                    trControl = trCtrl)

# Summaries
summary(LogAICFull) # GLM model info
LogAICFull # Caret model info
LogAICFull$finalModel


# Predictions Training Set
LogAICFull_pred <- predict(LogAICFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(LogAICFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Training Set
LogAICFull_ROC <- roc(fct_relevel(testing_dummied_fac_Miss$Comp_30, 'Comp'), 
                      predict(LogAICFull, testing_dummied_fac_Miss, type = 'prob')[,1],
                      ci = TRUE)
# ROC plot
plot.roc(LogAICFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(LogAICFull_ROC)








########################################################################
### 
###                           Decision Tree
### 
########################################################################

# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


set.seed(337)
dtFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                method = "rpart",
                metric = "ROC",
                trControl = trCtrl,
                tuneLength = 6)

# Summaries
dtFull # Model info
dtFull$finalModel # Final model



# Predictions Training Set
dtFull_pred <- predict(dtFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(dtFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix


# ROC Training Set
dtFull_ROC <- roc(testing_dummied_fac_Miss_cat$Comp_30, 
                  predict(dtFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1],
                  ci = TRUE)

# ROC plot
plot.roc(dtFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(dtFull_ROC) 





########################################################################
### 
###                                ADABoost
### 
########################################################################


# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


set.seed(337)
adaBoostFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                      training_dummied_fac_Miss_cat$Comp_30,
                      method = "adaboost",
                      metric = "ROC",
                      preProc = c("center", "scale"),
                      tuneLength = 4,
                      trControl = trCtrl)

# Summaries
adaBoostFull
adaBoostFull$finalModel


# Predictions Training Set
adaBoostFull_pred <- predict(adaBoostFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(adaBoostFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix


# ROC Training Set
adaBoostFull_ROC <- roc(testing_dummied_fac_Miss_cat$Comp_30, 
                        predict(adaBoostFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1],
                        ci = TRUE)

# ROC plot
plot.roc(adaBoostFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(adaBoostFull_ROC) 





########################################################################
### 
###                                XGBoost
### 
########################################################################


set.seed(337)
XGBoostFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     method = "xgbTree",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneLength = 6,
                     trControl = trCtrl)

# Summaries
XGBoostFull
XGBoostFull$finalModel


# Predictions Training Set
XGBoostFull_pred <- predict(XGBoostFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(XGBoostFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Training Set
XGBoostFull_ROC <- roc(testing_dummied_fac_Miss$Comp_30, 
                       predict(XGBoostFull, testing_dummied_fac_Miss, type = 'prob')[,1],
                       ci = TRUE)

# ROC plot
plot.roc(XGBoostFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(XGBoostFull_ROC)







########################################################################
### 
###                      SVM 
### 
########################################################################


set.seed(337)
svmFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                 training_dummied_fac_Miss$Comp_30,
                 method = "svmRadial",
                 metric = "ROC",
                 tuneLength = 6,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)
svmFull # Model info
svmFull$finalmodel # Final model


predict(svmFull, testing_dummied_fac_Miss)






########################################################################
### 
###                           Neural Network
### 
########################################################################


########################################################################
###  Average Neural Net

set.seed(337)
avNNetFull <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                    training_dummied_fac_Miss$Comp_30,
                    method = 'avNNet',
                    preProc = c("center", "scale"),
                    tuneLength = 4,
                    repeats = 10,
                    trace = FALSE, lineout = TRUE,
                    metric = 'ROC',
                    trControl = trCtrl)

# Summaries
avNNetFull # Model info
avNNetFull$finalModel


# Test set predictions
avNNetFull_pred <- predict(avNNetFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(avNNetFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Training Set
avNNetFull_ROC <- roc(testing_dummied_fac_Miss$Comp_30, 
                      predict(avNNetFull, testing_dummied_fac_Miss, type = 'prob')[,1])
# ROC plot
plot.roc(avNNetFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(avNNetFull_ROC) # AUC




