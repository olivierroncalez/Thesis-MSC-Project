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
data_facMiss <- as.data.frame(data_facMiss)
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
                     repeats = 3,
                     summaryFunction = fiveStats,
                     classProbs = TRUE,
                     index = dummy_index, # IMPORTANT! Uses the same index for all models
                     allowParallel = TRUE,
                     verboseIter = TRUE,
                     savePredictions = TRUE,
                     returnResamp = 'final')





########################################################################
### 
###                 Recursive Feature Elimination
### 
########################################################################


# Control object for implementing recursive feature elimination
rfCtrl <- rfeControl(method = "repeatedcv",
                         repeats = 3,
                         summaryFunction = fiveStats,
                         classProbs = TRUE,
                         index = dummy_index, # IMPORTANT! 
                         allowParallel = TRUE,
                         verboseIter = TRUE,
                     returnResamp = 'final',
                     saveDetails = TRUE)






########################################################################
### 
###                      Selection by Filter
### 
########################################################################


# Control object for implementing (needs "functions" argument)
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 5,
                      verbose = TRUE,
                      index = index_noMiss # IMPORTANT! 
                      )






# ML Models (Basic Implementation) --------------------------------------------------


########################################################################
### 
###                      Logistic Regression
### 
########################################################################




### Standard Logistic Regression

set.seed(337)
logisticFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                      fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), # Relevel factors to make 'healthy' the class considered. GLM uses 2nd class by default.
                      method = 'glm',
                      family = 'binomial',
                      preProcess = c('center', 'scale'),
                      trace = 0, # No verbose printout
                      trControl = trCtrl,
                      metric = 'ROC') # Maximize ROC metric

# Summaries
summary(logisticFull) # GLM model info
logisticFull # Caret model info
logisticFull$finalModel # Final model


# Predictions Training Set
logisticFull_pred <- predict(logisticFull, 
                             testing_dummied_fac_Miss, 
                             preProcess = c("center", "scale")) # Predicting test set

confusionMatrix(logisticFull_pred, 
                testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Training Set
logisticFull_ROC <- roc(fct_relevel(testing_dummied_fac_Miss$Comp_30, 'Comp'), 
    predict(logisticFull, testing_dummied_fac_Miss, type = 'prob')[,1])
# ROC plot
plot.roc(logisticFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(logisticFull_ROC) 







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
                        predict(LogAICFull, testing_dummied_fac_Miss, type = 'prob')[,1])
# ROC plot
plot.roc(LogAICFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(LogAICFull_ROC)







########################################################################
### 
###                           Random Forest
###
########################################################################


# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


# Grid of tuning parameters to try
rf_grid <- expand.grid(mtry = c(1:7))

set.seed(337)
rfFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                      method = "rf",
                      metric = "ROC",
                    preProc = c("center", "scale"),
                      tuneGrid = rf_grid,
                      ntree = 1000,
                      trControl = trCtrl)

# Summaries
rfFull # Model info
rfFull$finalModel # Final model



# Predictions Training Set
rfFull_pred <- predict(rfFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(rfFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix


# ROC Training Set
rfFull_ROC <- roc(testing_dummied_fac_Miss_cat$Comp_30, 
                        predict(rfFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1])

# ROC plot
plot.roc(rfFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(rfFull_ROC) 







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
                  predict(dtFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1])

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
                tuneLength = 6,
                trControl = trCtrl)








########################################################################
### 
###                      SVM (currently not working)
### 
########################################################################


# set.seed(337)
# svmFull <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
#                  training_dummied_fac_Miss$Comp_30,
#                  method = "svmRadial",
#                  metric = "ROC",
#                  tuneLength = 6,
#                  preProc = c("center", "scale"),
#                  trControl = trCtrl)
# svmFull # Model info
# svmFull$finalmodel # Final model
# 
# 
# predict(svmFull, testing_dummied_fac_Miss)





########################################################################
### 
###                                Naive Bayes
### 
########################################################################


# WARNING : The dataset used here is the exact same as those used in other models EXCEPT 
# for the fact that factor variables have been recoded as such. All indices and order are
# retained. The append of '_cat' symbolizes this.


set.seed(337)
nbFull <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                method = "nb",
                metric = "ROC",
                preProc = c("center", "scale"),
                trControl = trCtrl)

# Summaries
nbFull # Model info
nbFull$finalModel


# Predictions Training Set
nbFull_pred <- predict(nbFull, testing_dummied_fac_Miss_cat) # Predicting test set
confusionMatrix(nbFull_pred, testing_dummied_fac_Miss_cat$Comp_30) # Confusion matrix


# ROC Training Set
nbFull_ROC <- roc(testing_dummied_fac_Miss_cat$Comp_30, 
                  predict(nbFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1])

# ROC plot
plot.roc(nbFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(nbFull_ROC) 






########################################################################
### 
###                                kNN
### 
########################################################################


set.seed(337)
knnFull <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                 training_dummied_fac_Miss$Comp_30,
                 method = "kknn",
                 metric = "ROC",
                 tuneLength = 10,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)

# Summaries
knnFull # Model info
knnFull$finalModel


# Test Set predictions
knnFull_pred <- predict(knnFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(knnFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Test Set
knnFull_ROC <- roc(testing_dummied_fac_Miss$Comp_30, 
                  predict(knnFull, testing_dummied_fac_Miss, type = 'prob')[,1])

# ROC plot
plot.roc(knnFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(knnFull_ROC) 







########################################################################
### 
###                           Neural Network
### 
########################################################################


set.seed(337)
nnetFull <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  method = 'nnet',
                  tuneLength = 4,
                  preProc = c("center", "scale"),
                  trace = FALSE, lineout = TRUE,
                  metric = 'ROC',
                  trControl = trCtrl)

# Summaries
nnetFull # Model info
nnetFull$finalModel


# Test set predictions
nnetFull_pred <- predict(nnetFull, testing_dummied_fac_Miss) # Predicting test set
confusionMatrix(nnetFull_pred, testing_dummied_fac_Miss$Comp_30) # Confusion matrix


# ROC Training Set
nnetFull_ROC <- roc(testing_dummied_fac_Miss$Comp_30, 
                   predict(nnetFull, testing_dummied_fac_Miss, type = 'prob')[,1], 
                   ci = TRUE)

# ROC plot
plot.roc(nnetFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(nnetFull_ROC) # AUC






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







# Model Comparisons (Full) ----------------------------------------------------------------


########################################################################
### 
###                           Resamples
### 
########################################################################

# Resampling model list
model_list <- list('Logistic Reg' = logisticFull,
                   'Random Forest' = rfFull,
                   'SVM' = svmFull,
                   'N. Bayes' = nbFull,
                   'kNN' = knnFull,
                   'Neural Net' = nnetFull)
resample_list  <- resamples(model_list) 


# Summary
Full_resample_summary <- summary(resample_list)
Full_resample_summary # Explore the full summary for all 5 metrics
Full_resample_summary$statistics$ROC[, -7] # View ROC resampling results


# Inferential Statistics (pairwise comparisons) - ROC 
diff.resamples.Full <- diff(resample_list, metric = 'ROC', adjustment = 'none')
summary(diff.resamples.Full)



# Univariate inferential statistics
compare_models(rfFull, knnFull, metric = 'ROC') # Same as above is no adjustment is made
# Conducts one sample t-test using the difference scores of all resample folds (30 in this case)
# For accuracy comparisons, binomial test is used by the caret package.




########################################################################
### 
###                      Test set predictions
### 
########################################################################


# Test set predictions
logisticFull_pred
rfFull_pred
nbFull_pred
knnFull_pred
nnetFull_pred


# Univariate inferential statistics (conducted on ROC)
roc.test(logisticFull_ROC, rfFull_ROC)


# AUC
auc(logisticFull_ROC)
auc(rfFull_ROC)
auc(nbFull_ROC)
auc(knnFull_ROC)
auc(nnetFull_ROC)


# AUC CI
ci.auc(logisticFull_ROC) 
ci.auc(rfFull_ROC)
ci.auc(nbFull_ROC)
ci.auc(knnFull_ROC)
ci.auc(nnetFull_ROC)





########################################################################
### 
###                           Visualizations
### 
########################################################################


########################################################################
### RESAMPLING


# Resampling results (inferential pairwise comparisons)
dotplot(diff.resamples.Full)
densityplot(diff.resamples.Full,
            metric = "ROC",
            auto.key = TRUE,
            pch = "|")


# Resampling results (no comparisons)
dotplot(resample_list , metric = 'ROC', axes = TRUE)
xyplot(resample_list , metric = 'ROC', axes = TRUE)





########################################################################
### TEST SET

# ROC Curves (full training data - holdout set) 
plot(logisticFull_ROC, legacy.axes = TRUE, print.thres = TRUE, main = 'ROC', identity.lty = 2)
plot(rfFull_ROC, legacy.axes = TRUE, print.thres = TRUE, add = TRUE, col = 2)
plot(nbFull_ROC, legacy.axes = TRUE, print.thres = TRUE, add = TRUE, col = 3)
plot(knnFull_ROC, legacy.axes = TRUE, print.thres = TRUE, add = TRUE, col = 4)
plot(nnetFull_ROC, legacy.axes = TRUE, print.thres = TRUE, add = TRUE, col = 5)
# Important to note that these models were trained on the full training data and are thus
# different to those in the resampled results

