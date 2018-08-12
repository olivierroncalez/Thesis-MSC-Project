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



# Turn off sampling
trCtrl$sampling <- NULL




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










########################################################################################################
########################################################################################################

#                                        Feature Selection

########################################################################################################
########################################################################################################



##############################################
#
#                   Functions 
#
##############################################


##############################################
# Decision Tree

dt_filter <- function(score, x, y) {
     keepers <- score > 0
     keepers
}


dt_score <- function(x, y) {
     dat <- dplyr::bind_cols(x, y = y)
     out <- rpart::rpart(y ~., data = dat)$variable.importance
     pred_names <- names(x)
     v = vector()
     for (i in pred_names) {
          if (i %in% names(out)) {
               v[i] <- out[i]
          } else {
               v[i] <- 0
          }
     }
     out <- v
     out
}





##############################################
# Univariate Filter

uni_pscore <- function(x, y) {
     
     numX <- length(unique(x))
     if(numX > 2) {
          out <- t.test(x ~ y)$p.value
     } else {
          out <- fisher.test(factor(x), y)$p.value
     }
     out
}


uni_pscore_filter <- function(score, x, y) {
     keepers <- (score <= 0.05)
     keepers
}









##############################################
#
#                   Control Objects
#
##############################################


# Five stats summary
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# SBF Control (must be the same as previous train control for fair comparison)
sbfCtrl <- sbfControl(method = "repeatedcv",
                      repeats = 3,
                      index = dummy_index, # IMPORTANT!
                      verbose = TRUE,
                      saveDetails = TRUE,
                      allowParallel = TRUE,
                      returnResamp = 'final')


# RFE Control (must be the same as previous train control for fair comparison)
rfeCtrl <- rfeControl(method = "repeatedcv", 
                      repeats = 3,
                      index = dummy_index,
                      saveDetails = TRUE,
                      returnResamp = 'final',
                      allowParallel = TRUE,
                      verbose = TRUE)


# Train control (if tuning is needed, simple 10-fold CV) 
trCtrl <-  trainControl(method = "cv",
                        classProbs = TRUE, # For maximizing ROC curve
                        verboseIter = TRUE,
                        allowParallel = FALSE,
                        summaryFunction = fiveStats)





##############################################
#
#               CaretSBF Objects
#
##############################################


# Decision Tree Embedded

dtFilter <- caretSBF
dtFilter$summary <- fiveStats
dtFilter$score <- dt_score
dtFilter$filter <- dt_filter





# Univariate Filter (t-test)

uni_filter <- caretSBF
uni_filter$summary <- fiveStats
uni_filter$score <- uni_pscore
uni_filter$filter <- uni_pscore_filter






############################################################################################
############################################################################################
#
#                                       SBF Models
#
############################################################################################
############################################################################################




###############################################
### 
###                 Logistic
###
###############################################


## UNIVARIATE FILTER


# Set control elements for univariate filter
sbfCtrl$multivariate <- FALSE
sbfCtrl$functions <- uni_filter


set.seed(337)
log_sbf_UNI <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                   fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                   sbfControl = sbfCtrl,
                   # Train elements below here
                   method = 'glm',
                   family = 'binomial',
                   preProcess = c('center', 'scale'))
log_sbf_UNI
log_sbf_UNI$fit




## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter

set.seed(337)
log_sbf_DT <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                  fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                  sbfControl = sbfCtrl,
                  # Train elements below here
                  method = 'glm',
                  family = 'binomial',
                  preProcess = c('center', 'scale'))
log_sbf_DT
log_sbf_DT$fit







###############################################
### 
###                 Random Forest (Works if data is not factor data (i.e. "_cat"))
###
###############################################


## UNIVARIATE FILTER


# Set control elements for univariate filter
sbfCtrl$multivariate <- FALSE
sbfCtrl$functions <- uni_filter

# Grid of tuning parameters to try
rf_grid <- expand.grid(mtry = c(1:7))

set.seed(337)
rf_sbf_UNI <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  trControl = trCtrl,
                  sbfControl = sbfCtrl,
                  # Train elements below here
                  tuneGrid = rf_grid,
                  metric = "ROC",
                  preProc = c("center", "scale"))
rf_sbf_UNI
rf_sbf_UNI$fit




## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter


set.seed(337)
rf_sbf_DT <- sbf(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                  training_dummied_fac_Miss_cat$Comp_30,
                  trControl = trCtrl,
                  sbfControl = sbfCtrl,
                  # Train elements below here
                  tuneGrid = rf_grid,
                  metric = "ROC",
                  preProc = c("center", "scale"))
rf_sbf_DT
rf_sbf_DT$fit





###############################################
### 
###                 Naive Bayes
###
###############################################


## UNIVARIATE FILTER


# Set control elements for univariate filter
sbfCtrl$multivariate <- FALSE
sbfCtrl$functions <- uni_filter


set.seed(337)
nb_sbf_UNI <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      trControl = trCtrl,
                      sbfControl = sbfCtrl,
                      # Train elements below here
                      method = "nb",
                      metric = "ROC",
                      preProc = c("center", "scale"))
nb_sbf_UNI
nb_sbf_UNI$fit


## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter


set.seed(337)
nb_sbf_DT <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     trControl = trCtrl,
                     sbfControl = sbfCtrl,
                     # Train elements below here
                     method = "nb",
                     metric = "ROC",
                     preProc = c("center", "scale"))
nb_sbf_DT
nb_sbf_DT$fit





###############################################
### 
###                 kNN
###
###############################################


## UNIVARIATE FILTER


# Set control elements for univariate filter
sbfCtrl$multivariate <- FALSE
sbfCtrl$functions <- uni_filter


set.seed(337)
knn_sbf_UNI <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                   training_dummied_fac_Miss$Comp_30,
                   trControl = trCtrl,
                   sbfControl = sbfCtrl,
                   # Train elements below here
                   method = "kknn",
                   metric = "ROC",
                   tuneLength = 10,
                   preProc = c("center", "scale"))
knn_sbf_UNI
knn_sbf_UNI$fit



## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter


set.seed(337)
knn_sbf_DT <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  trControl = trCtrl,
                  sbfControl = sbfCtrl,
                  # Train elements below here
                  method = "kknn",
                  metric = "ROC",
                  tuneLength = 10,
                  preProc = c("center", "scale"))
knn_sbf_DT
knn_sbf_DT$fit






###############################################
### 
###                 Neural Network
###
###############################################


## UNIVARIATE FILTER


# Set control elements for univariate filter
sbfCtrl$multivariate <- FALSE
sbfCtrl$functions <- uni_filter


set.seed(337)
nnet_sbf_UNI <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      trControl = trCtrl,
                      sbfControl = sbfCtrl,
                      # Train elements below here
                      method = 'nnet',
                      preProc = c("center", "scale"),
                      tuneLength = 4,
                      trace = FALSE, lineout = TRUE,
                      metric = 'ROC')
nnet_sbf_UNI
nnet_sbf_UNI$fit




## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter

set.seed(337)
nnet_sbf_DT <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     trControl = trCtrl,
                     sbfControl = sbfCtrl,
                     # Train elements below here
                     method = 'nnet',
                     preProc = c("center", "scale"),
                     tuneLength = 4,
                     trace = FALSE, lineout = TRUE,
                     metric = 'ROC')
nnet_sbf_DT
nnet_sbf_DT$fit







############################################################################################
############################################################################################
#
#                                       RFE Models
#
############################################################################################
############################################################################################


### Miscellaneous functions and control elements

# Sequence length for RFE
varSeq <- seq(1, length(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)])-1, by = 2)



###############################################
### 
###                 Logistic 
###
###############################################


rfeCtrl$functions <- lrFuncs
rfeCtrl$functions$summary <- fiveStats

# Change names to circumvent bug
var_names <- names(training_dummied_fac_Miss)
names(training_dummied_fac_Miss) <- gsub(pattern = "\\(|\\)",
                                         replacement = '',
                                         x = names(training_dummied_fac_Miss))


set.seed(337)
log_RFE <- rfe(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
               fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'),
               sizes = varSeq,
               metric = 'ROC',
               preProcess = c('center', 'scale'),
               rfeControl = rfeCtrl)
log_RFE






###############################################
### 
###                 Random Forest
###
###############################################


rfeCtrl$functions <- rfFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(337)
rf_RFE <- rfe(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
              training_dummied_fac_Miss_cat$Comp_30,
              rfeControl = rfeCtrl,
              metric = "ROC",
              preProc = c("center", "scale"),
              ntree = 1000,
              sizes = varSeq)
rf_RFE
rf_RFE$fit






###############################################
### 
###                 Naive Bayes
###
###############################################


rfeCtrl$functions <- nbFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(337)
nb_RFE <- rfe(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  rfeControl = rfeCtrl,
                  sizes = varSeq,
                  metric = "ROC",
                  preProc = c("center", "scale"))
nb_RFE






###############################################
### 
###                   kNN 
###
###############################################


rfeCtrl$functions <- caretFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(721)
knn_RFE <- rfe(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
               training_dummied_fac_Miss$Comp_30,
               sizes = varSeq,
               metric = "ROC",
               method = "kknn",
               tuneLength = 10,
               preProc = c("center", "scale"),
               trControl = trCtrl,
               rfeControl = rfeCtrl)
knn_RFE




###############################################
### 
###                Nnet 
###
###############################################

rfeCtrl$functions <- caretFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(337)
nnet_RFE <- rfe(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                  training_dummied_fac_Miss$Comp_30,
                  sizes = varSeq,
                  method = 'nnet',
                  preProc = c("center", "scale"),
                  tuneLength = 4,
                  trace = FALSE, 
                  lineout = TRUE,
                  metric = 'ROC',
                  trControl = trCtrl,
                  rfeControl = rfeCtrl)
nnet_RFE







# Model Comparisons (Full) ----------------------------------------------------------------

# load('results_ML_Models_Msc_final3.RData') # Load the models which have been run
# Final '3' adds some of the resamples lists from version 1. No additional models were run.

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
log_sbf_UNI
log_sbf_DT
log_RFE

# Test set results
logisticFull_ROC # Test set ROC
logisticFull_pred # Test set predictions





### Random Forests

# Resampling results
rfFull
rfFull_up
rfFull_down
rfFull_smote
rf_sbf_UNI
rf_sbf_DT
rf_RFE


# Test set results
rfFull_ROC # Test set ROC
rfFull_pred # Test set predictions





### Naive Bayes

# Resampling results
nbFull
nbFull_up
nbFull_down
nbFull_smote
nb_sbf_UNI
nb_sbf_DT
nb_RFE



# Test set results
nbFull_ROC # Test set ROC
nbFull_pred # Test set predictions




### K-Nearest Neighbor

# Resampling results
knnFull
knnFull_up
knnFull_down
knnFull_smote
knn_sbf_UNI
knn_sbf_DT
knn_RFE

# Test set results
knnFull_ROC # Test set ROC
knnFull_pred # Test set predictions




### Neural Net

# Resampling results
nnetFull
nnetFull_up
nnetFull_down
nnetFull_smote
nnet_sbf_UNI
nnet_sbf_DT
nnet_RFE



# Test set results
nnetFull_ROC # Test set ROC
nnetFull_pred # Test set predictions


########################################################################
### 
###                           Resamples
### 
########################################################################


########################################################################
### Model Lists


### Resampling model list (full)

model_list <- list('Logistic Reg' = logisticFull,
                   'Random Forest' = rfFull,
                   'N. Bayes' = nbFull,
                   'kNN' = knnFull,
                   'Neural Net' = nnetFull)
resample_list  <- resamples(model_list) 
summary(resample_list)


### Resampling model list (SBF uni)

model_list_sbf_UNI <- list('Logistic Reg' = log_sbf_UNI,
                           'Random Forest' = rf_sbf_UNI,
                           'N. Bayes' = nb_sbf_UNI,
                           'kNN' = knn_sbf_UNI,
                           'Neural Net' = nnet_sbf_UNI)
resample_list_sbf_UNI  <- resamples(model_list_sbf_UNI) 
summary(resample_list_sbf_UNI)




### Resampling model list (SBF DT)

model_list_sbf_DT <- list('Logistic Reg' = log_sbf_DT,
                          'Random Forest' = rf_sbf_DT,
                          'N. Bayes' = nb_sbf_DT,
                          'kNN' = knn_sbf_DT,
                          'Neural Net' = nnet_sbf_DT)
resample_list_sbf_DT  <- resamples(model_list_sbf_DT) 
summary(resample_list_sbf_DT)



### Resampling model list (RFE)

model_list_RFE <- list('Logistic Reg' = log_RFE,
                   'Random Forest' = rf_RFE,
                   'N. Bayes' = nb_RFE,
                   'kNN' = knn_RFE,
                   'Neural Net' = nnet_RFE)
resample_list_RFE  <- resamples(model_list_RFE) 
summary(resample_list_RFE)



### Resampling model list (Sampling)

model_list_sample <- list('Logistic Reg (up)' = logisticFull_up,
                       'Logistic Reg (down)' = logisticFull_down,
                       'Logistic Reg (SMOTE)' = logisticFull_smote,
                       'Random Forest (up)' = rfFull_up,
                       'Random Forest (down)' = rfFull_down,
                       'Random Forest (SMOTE)' = rfFull_smote,
                       'N. Bayes (up)' = nbFull_up,
                       'N. Bayes (down)' = nbFull_down,
                       'N. Bayes (SMOTE)' = nbFull_smote,
                       'kNN (up)' = knnFull_up,
                       'kNN (down)' = knnFull_down,
                       'kNN (SMOTE)' = knnFull_smote,
                       'Neural Net (up)' = nnetFull_up,
                       'Neural Net (down)' = nnetFull_down,
                       'Neural Net (SMOTE)' = nnetFull_smote)
resample_list_sample  <- resamples(model_list_sample) 
summary(resample_list_sample)



### Resampling model list (BEST)
model_list_best <- list('Logistic Reg (Baseline)' = logisticFull,
                        'Logistic Reg (RFE)' = log_RFE,
                        'Random Forest (Uni)' = rf_sbf_UNI,
                        'N. Bayes (RFE)' = nb_RFE,
                        'kNN (RFE)' = knn_RFE,
                        'Neural Net (DT)' = nnet_sbf_DT)
resample_list_best <- resamples(model_list_best)
summary(resample_list_best)




########################################################################
### Summaries


# Summary (full)
resample_summary_Full <- summary(resample_list)
resample_summary_Full # Explore the full summary for all 5 metrics
resample_summary_Full$statistics$ROC[, -7] # View ROC resampling results
# xtable(resample_summary_Full$statistics$ROC[, -7], caption = 'Resample ROC Summary Statistics', digits = 3)



# Summary (SBF uni)
resample_summary_sbf_UNI <- summary(resample_list_sbf_UNI)
resample_summary_sbf_UNI # Explore the full summary for all 5 metrics
resample_summary_sbf_UNI$statistics$ROC[, -7] # View ROC resampling results
# xtable(resample_summary_sbf_UNI$statistics$ROC[, -7])



# Summary (SBF DT)
resample_summary_sbf_DT <- summary(resample_list_sbf_DT)
resample_summary_sbf_DT # Explore the full summary for all 5 metrics
resample_summary_sbf_DT$statistics$ROC[, -7] # View ROC resampling results
# xtable(resample_summary_sbf_DT$statistics$ROC[, -7])



# Summary (RFE)
resample_summary_RFE <- summary(resample_list_RFE)
resample_summary_RFE # Explore the full summary for all 5 metrics
resample_summary_RFE$statistics$ROC[, -7] # View ROC resampling results
# xtable(resample_summary_RFE$statistics$ROC[, -7])



# Summary (Sampling)
resample_summary_sample <- summary(resample_list_sample)
resample_summary_sample # Explore the full summary for all 5 metrics
resample_summary_sample$statistics$ROC[, -7] # View ROC resampling results
xtable(resample_summary_sample$statistics$ROC[, -7])



# Summary (Best)
resample_summary_best <- summary(resample_list_best)
resample_summary_best
resample_summary_best$statistics$ROC[, -7]
xtable(resample_summary_best$statistics$ROC[, -7])





########################################################################
### Inferential Statistics



# Inferential Statistics (pairwise comparisons) - ROC Full predictors
diff.resamples.Full <- diff(resample_list, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
diff.resamples.Full.Summary <- summary(diff.resamples.Full, digits = 1) 
diff.resamples.Full.Summary
# diff.resamples.Full.Summary$table$ROC %>% xtable(digits = 1)


# Inferential Statistics (pairwise comparisons) - ROC SBF uni predictors
diff.resamples.sbf.UNI <- diff(resample_list_sbf_UNI, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
diff.resamples.sbf.UNI.Summary <- summary(diff.resamples.sbf.UNI, digits = 1) 
diff.resamples.sbf.UNI.Summary
# diff.resamples.sbf.UNI.Summary$table$ROC %>% xtable(digits = 1)


# Inferential Statistics (pairwise comparisons) - ROC SBF dt predictors
diff.resamples.sbf.DT <- diff(resample_list_sbf_DT, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
diff.resamples.sbf.DT.Summary <- summary(diff.resamples.sbf.DT, digits = 1) 
diff.resamples.sbf.DT.Summary
# diff.resamples.sbf.DT.Summary$table$ROC %>% xtable(digits = 1)


# Inferential Statistics (pairwise comparisons) - ROC RFE predictors
diff.resamples.rfe <- diff(resample_list_RFE, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
diff.resamples.rfe.Summary <- summary(diff.resamples.rfe, digits = 1) 
diff.resamples.rfe.Summary
# diff.resamples.rfe.Summary$table$ROC %>% xtable(digits = 1)


# Inferential Statistics (pairwise comparisons) - ROC sampling predictors
diff.resamples.sampling <- diff(resample_list_sample, metric = 'ROC', adjustment = 'none') # No pairwise adjustment
diff.resamples.sampling.Summary <- summary(diff.resamples.sampling, digits = 1) 
diff.resamples.sampling.Summary
# diff.resamples.sampling.Summary$table$ROC %>% xtable(digits = 1)



# Inferential Statistics (pairwise comparisons) 
diff.resamples.best <- diff(resample_list_best, metric = 'ROC', adjustment = 'none')
diff.resmaples.best.Summary <- summary(diff.resamples.best, digits = 1)
diff.resmaples.best.Summary
# diff.resmaples.best.Summary$table$ROC %>% xtable(digits = 1)




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
model_pred_full <- bind_cols('Logistic Reg' = logisticFull_pred, 
                             'Random Forest' = rfFull_pred,
                             'N. Bayes' = nbFull_pred,
                             'kNN' = knnFull_pred,
                             'Neural Net' = nnetFull_pred)
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

# Test set table
test.auc.full <- bind_cols(logFull_ROC = as.numeric(ci.auc(logisticFull_ROC)),
          rfFull_ROC = as.numeric(ci.auc(rfFull_ROC)),
          nbFull_ROC = as.numeric(ci.auc(nbFull_ROC)),
          knnFull_ROC = as.numeric(ci.auc(knnFull_ROC)),
          nnet_Full_ROC = as.numeric(ci.auc(nnetFull_ROC)))







########################################################################
### 
###                           Visualizations
### 
########################################################################





########################################################################
### RESAMPLING


# Resampling results (no comparisons)
dotplot(resample_list, metric = 'ROC', axes = TRUE) # Rplot1 Full
dotplot(resample_list_sbf_UNI, metric = 'ROC')
dotplot(resample_list_sbf_DT, metric = 'ROC')
dotplot(resample_list_RFE, metric = 'ROC')
dotplot(resample_list_sample, metric = 'ROC')
dotplot(resample_list_best, metric = 'ROC')






# Resampling results (inferential pairwise comparisons)
dotplot(diff.resamples.Full) # Rplot2 Full Diff
dotplot(diff.resamples.rfe)
dotplot(diff.resamples.best)
dotplot(diff.resamples.sbf.UNI)
dotplot(diff.resamples.sbf.DT)





# Resampling results (RFE) model performance plots

LR_RFE_plot <- ggplot(log_RFE) + labs(y = '', title = 'Logistic Reg.',
                                      x = '') + theme(plot.title = element_text(hjust = 0.5))
RF_RFE_plot <- ggplot(rf_RFE) + labs(y = '', title = 'Random Forest',
                                     x = '') + theme(plot.title = element_text(hjust = 0.5))
knn_RFE_plot <- ggplot(knn_RFE) + labs(y = '', title = 'kNN',
                                       x = '') + theme(plot.title = element_text(hjust = 0.5))
nb_RFE_plot <- ggplot(nb_RFE) + labs(y = '', title = 'N. Bayes',
                                     x = '') + theme(plot.title = element_text(hjust = 0.5))
nnet_RFE_plot <- ggplot(nnet_RFE) + labs(y = '', title = 'Neural Net',
                                         x = '') + theme(plot.title = element_text(hjust = 0.5))


library(gridExtra)
library(grid)
grid.arrange(LR_RFE_plot, RF_RFE_plot, knn_RFE_plot,
             nb_RFE_plot, nnet_RFE_plot, nrow = 3, ncol = 2,
             bottom = textGrob('Number of Variables', gp = gpar(fontface = 'bold', fontsize = '14')), 
             left = textGrob('ROC', gp = gpar(fontface = 'bold', fontsize = '14'), rot = 90))




# View tuning parameters over resampling results
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








