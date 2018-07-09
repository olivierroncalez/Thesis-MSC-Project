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
trCtrl <- trainControl(method = "repeatedcv", # Defaults to 10-CV
                     repeats = 3,
                     summaryFunction = fiveStats,
                     index = dummy_index, # IMPORTANT! Uses the same index for all models
                     classProbs = TRUE,
                     allowParallel = TRUE,
                     verboseIter = TRUE,
                     savePredictions = TRUE,
                     returnResamp = 'final')






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
                      trControl = trCtrl) 



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
    predict(logisticFull, testing_dummied_fac_Miss, type = 'prob')[,1],
    ci = TRUE)
# ROC plot
plot.roc(logisticFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(logisticFull_ROC) 





########################################################################
### Sampling


### UP SAMPLING
trCtrl$sampling <- 'up'

set.seed(337)
logisticFull_up <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], 
                      fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                      method = 'glm',
                      family = 'binomial',
                      preProcess = c('center', 'scale'),
                      trace = 0, 
                      trControl = trCtrl) 
logisticFull_up


### DOWN SAMPLING
trCtrl$sampling <- 'down'

set.seed(337)
logisticFull_down <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], 
                         fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                         method = 'glm',
                         family = 'binomial',
                         preProcess = c('center', 'scale'),
                         trace = 0, 
                         trControl = trCtrl) 
logisticFull_down


### SMOTE SAMPLING
trCtrl$sampling <- 'smote'

set.seed(337)
logisticFull_smote <- train(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], 
                           fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                           method = 'glm',
                           family = 'binomial',
                           preProcess = c('center', 'scale'),
                           trace = 0, 
                           trControl = trCtrl) 
logisticFull_smote


# Turn off sampling
trCtrl$sampling <- NULL









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
                        predict(rfFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1],
                  ci = TRUE)

# ROC plot
plot.roc(rfFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(rfFull_ROC) 






########################################################################
### Sampling


### UP SAMPLING
trCtrl$sampling <- 'up'


set.seed(337)
rfFull_up <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                method = "rf",
                metric = "ROC",
                preProc = c("center", "scale"),
                tuneGrid = rf_grid,
                ntree = 1000,
                trControl = trCtrl)
rfFull_up


### DOWN SAMPLING
trCtrl$sampling <- 'down'


set.seed(337)
rfFull_down <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                   training_dummied_fac_Miss_cat$Comp_30,
                   method = "rf",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   tuneGrid = rf_grid,
                   ntree = 1000,
                   trControl = trCtrl)
rfFull_down


### SMOTE SAMPLING
trCtrl$sampling <- 'smote'


set.seed(337)
rfFull_smote <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                     training_dummied_fac_Miss_cat$Comp_30,
                     method = "rf",
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     tuneGrid = rf_grid,
                     ntree = 1000,
                     trControl = trCtrl)
rfFull_smote


# Turn off sampling
trCtrl$sampling <- NULL






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
                  predict(nbFull, testing_dummied_fac_Miss_cat, type = 'prob')[,1],
                  ci = TRUE)

# ROC plot
plot.roc(nbFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(nbFull_ROC) 





########################################################################
### Sampling


### UP SAMPLING
trCtrl$sampling <- 'up'


set.seed(337)
nbFull_up <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                training_dummied_fac_Miss_cat$Comp_30,
                method = "nb",
                metric = "ROC",
                preProc = c("center", "scale"),
                trControl = trCtrl)
nbFull_up


### DOWN SAMPLING
trCtrl$sampling <- 'down'


set.seed(337)
nbFull_down <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                   training_dummied_fac_Miss_cat$Comp_30,
                   method = "nb",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = trCtrl)
nbFull_down


### SMOTE SAMPLING
trCtrl$sampling <- 'smote'


set.seed(337)
nbFull_smote <- train(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                   training_dummied_fac_Miss_cat$Comp_30,
                   method = "nb",
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = trCtrl)
nbFull_smote


# Turn off sampling
trCtrl$sampling <- NULL






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
                  predict(knnFull, testing_dummied_fac_Miss, type = 'prob')[,1],
                  ci = TRUE)

# ROC plot
plot.roc(knnFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(knnFull_ROC) 





########################################################################
### Sampling


### UP SAMPLING
trCtrl$sampling <- 'up'

set.seed(337)
knnFull_up <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                 training_dummied_fac_Miss$Comp_30,
                 method = "kknn",
                 metric = "ROC",
                 tuneLength = 10,
                 preProc = c("center", "scale"),
                 trControl = trCtrl)
knnFull_up


### DOWN SAMPLING
trCtrl$sampling <- 'down'

set.seed(337)
knnFull_down <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                    training_dummied_fac_Miss$Comp_30,
                    method = "kknn",
                    metric = "ROC",
                    tuneLength = 10,
                    preProc = c("center", "scale"),
                    trControl = trCtrl)
knnFull_down


### SMOTE SAMPLING
trCtrl$sampling <- 'smote'

set.seed(337)
knnFull_smote <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      method = "kknn",
                      metric = "ROC",
                      tuneLength = 10,
                      preProc = c("center", "scale"),
                      trControl = trCtrl)
knnFull_smote






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


# ROC Test Set
nnetFull_ROC <- roc(testing_dummied_fac_Miss$Comp_30, 
                   predict(nnetFull, testing_dummied_fac_Miss, type = 'prob')[,1], 
                   ci = TRUE)

# ROC plot
plot.roc(nnetFull_ROC, legacy.axes = TRUE, print.thres = TRUE) # Plotting
# AUC
auc(nnetFull_ROC) # AUC





########################################################################
### Sampling


### UP SAMPLING
trCtrl$sampling <- 'up'


set.seed(337)
nnetFull_up <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  method = 'nnet',
                  tuneLength = 4,
                  preProc = c("center", "scale"),
                  trace = FALSE, lineout = TRUE,
                  metric = 'ROC',
                  trControl = trCtrl)
nnetFull_up


### DOWN SAMPLING
trCtrl$sampling <- 'down'


set.seed(337)
nnetFull_down <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     method = 'nnet',
                     tuneLength = 4,
                     preProc = c("center", "scale"),
                     trace = FALSE, lineout = TRUE,
                     metric = 'ROC',
                     trControl = trCtrl)
nnetFull_down


### SMOTE SAMPLING
trCtrl$sampling <- 'smote'


set.seed(337)
nnetFull_smote <- train(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                       training_dummied_fac_Miss$Comp_30,
                       method = 'nnet',
                       tuneLength = 4,
                       preProc = c("center", "scale"),
                       trace = FALSE, lineout = TRUE,
                       metric = 'ROC',
                       trControl = trCtrl)
nnetFull_smote


trCtrl$sampling <- NULL






# Model Comparisons (Full) ----------------------------------------------------------------


########################################################################
### 
###                           Result Objects
### 
########################################################################

### Logistic Regression

# Resampling results
logisticFull
logisticFull_up
logisticFull_down
logisticFull_smote

# Test set results
logisticFull_ROC # Test set ROC
logisticFull_pred # Test set predictions




### Random Forests

# Resampling results
rfFull
rfFull_up
rfFull_down
rfFull_smote

# Test set results
rfFull_ROC # Test set ROC
rfFull_pred # Test set predictions




### Naive Bayes

# Resampling results
knnFull
knnFull_up
knnFull_down
knnFull_smote

# Test set results
knnFull_ROC # Test set ROC
knnFull_pred # Test set predictions




### K-Nearest Neighbor

# Resampling results
knnFull
knnFull_up
knnFull_down
knnFull_smote

# Test set results
knnFull_ROC # Test set ROC
knnFull_pred # Test set predictions




### Neural Net

# Resampling results
nnetFull
nnetFull_up
nnetFull_down
nnetFull_smote

# Test set results
nnetFull_ROC # Test set ROC
nnetFull_pred # Test set predictions


########################################################################
### 
###                           Resamples
### 
########################################################################

# Resampling model list
model_list <- list('Logistic Reg' = logisticFull,
                   'Random Forest' = rfFull,
                   'N. Bayes' = nbFull,
                   'kNN' = knnFull,
                   'Neural Net' = nnetFull)
resample_list  <- resamples(model_list) 


# Summary
Full_resample_summary <- summary(resample_list)
Full_resample_summary # Explore the full summary for all 5 metrics
Full_resample_summary$statistics$ROC[, -7] # View ROC resampling results


# Inferential Statistics (pairwise comparisons) - ROC 
diff.resamples.Full <- diff(resample_list, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
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


# Cohen's Kappa
model_pred_full <- bind_cols(logFull = logisticFull_pred, 
          rfFull = rfFull_pred,
          nbFull = nbFull_pred,
          knnFull = knnFull_pred,
          nnetFull = nnetFull_pred)
psych::cohen.kappa(model_pred_full, alpha = 0.05)


# Univariate inferential statistics (conducted on ROC)
roc.test(logisticFull_ROC, rfFull_ROC) 
roc.test(logisticFull_ROC, nbFull_ROC)
roc.test(logisticFull_ROC, knnFull_ROC)
roc.test(logisticFull_ROC, nnetFull_ROC)
roc.test(rfFull_ROC, nbFull_ROC)
roc.test(rfFull_ROC, knnFull_ROC)
roc.test(rfFull_ROC, nnetFull_ROC)
roc.test(nbFull_ROC, knnFull_ROC)
roc.test(nbFull_ROC, nnetFull_ROC)
roc.test(knnFull_ROC, nnetFull_ROC)


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


# Resampling results (no comparisons)
dotplot(resample_list , metric = 'ROC', axes = TRUE)


# Resampling results (inferential pairwise comparisons)
dotplot(diff.resamples.Full)


# View tuning parameters over resamplign results
plot(rfFull)
plot(nbFull)
plot(knnFull)
plot(nnetFull)




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











########################################################################################################
########################################################################################################

#                                        Feature Selection

########################################################################################################
########################################################################################################



########################################################################
### 
###                      Embedded
### 
########################################################################

