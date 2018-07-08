


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
# Relief Filter 

CORElearn::attrEval(Comp_30 ~., 
                    data = training_dummied_fac_Miss,
                    estimator = 'GainRatio')







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
                   index = index,
                   saveDetails = TRUE,
                   returnResamp = "final",
                   allowParallel = TRUE,
                   verbose = TRUE)





# Train control (if tuning is needed, simple 5-fold CV) - Don't forget to include ROC metric
trCtrl <-  trainControl(method = "cv",
                        classProbs = TRUE, # For maximizing ROC curve
                        verboseIter = TRUE,
                        allowParallel = TRUE)




##############################################
#
#                   CaretSBF
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
              method = 'glm',
              family = 'binomial',
              preProcess = c('center', 'scale'),
              sbfControl = sbfCtrl,
              trControl = trCtrl)
log_sbf_UNI
log_sbf_UNI$fit




## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter

set.seed(337)
log_sbf_DT <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)], # Last column is target
                   fct_relevel(training_dummied_fac_Miss$Comp_30, 'Comp'), 
                   method = 'glm',
                   family = 'binomial',
                   preProcess = c('center', 'scale'),
                   sbfControl = sbfCtrl)
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
rf_sbf_UNI <- sbf(training_dummied_fac_Miss_cat[, -ncol(training_dummied_fac_Miss_cat)],
                  training_dummied_fac_Miss_cat$Comp_30,
                  trControl = trCtrl,
                  sbfControl = sbfCtrl,
                  tuneGrid = rf_grid,
                  metric = "ROC",
                  preProc = c("center", "scale"))
rf_sbf_UNI
rf_sbf_UNI$fit





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
nbFull_sbf_UNI <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                training_dummied_fac_Miss$Comp_30,
                trControl = trCtrl,
                sbfControl = sbfCtrl,
                # Train elements below here
                method = "nb",
                metric = "ROC",
                preProc = c("center", "scale"))
nbFull_sbf_UNI
nbFull_sbf_UNI$fit


## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter


set.seed(337)
nbFull_sbf_DT <- sbf(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      trControl = trCtrl,
                      sbfControl = sbfCtrl,
                      # Train elements below here
                      method = "nb",
                      metric = "ROC",
                      preProc = c("center", "scale"))
nbFull_sbf_DT
nbFull_sbf_DT$fit




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
avNNet_sbf_UNI <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      method = 'avNNet',
                      preProc = c("center", "scale"),
                      tuneLength = 4,
                      trace = FALSE, lineout = TRUE,
                      metric = 'ROC',
                      trControl = trCtrl,
                      sbfControl = sbfCtrl)
avNNet_sbf_UNI
avNNet_sbf_UNI$fit




## EMBEDDED FILTER

# Set control elements for dt filter
sbfCtrl$multivariate <- TRUE
sbfCtrl$functions <- dtFilter

set.seed(337)
avNNet_sbf_DT <- sbf(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     method = 'avNNet',
                     preProc = c("center", "scale"),
                     tuneLength = 4,
                     trace = FALSE, lineout = TRUE,
                     metric = 'ROC',
                     trControl = trCtrl,
                     sbfControl = sbfCtrl)
avNNet_sbf_DT
avNNet_sbf_DT$fit







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
###                 Logistic (Error message)
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
               training_dummied_fac_Miss$Comp_30,
               sizes = varSeq,
               metric = 'ROC',
               preProcess = c('center', 'scale'),
               rfeControl = rfeCtrl)







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
rf_RFE$fit






###############################################
### 
###                 Naive Bayes
###
###############################################


rfeCtrl$functions <- nbFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(337)
nbFull_RFE <- rfe(training_dummied_fac_Miss[, -ncol(training_dummied_fac_Miss)],
                      training_dummied_fac_Miss$Comp_30,
                      rfeControl = rfeCtrl,
                      sizes = varSeq,
                      metric = "ROC",
                      preProc = c("center", "scale"))
nbFull_RFE






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
               rfeControl = ctrl)





###############################################
### 
###                Nnet 
###
###############################################

rfeCtrl$functions <- caretFuncs
rfeCtrl$functions$summary <- fiveStats


set.seed(337)
avNNet_RFE <- rfe(training_dummied_fac_Miss[ , -ncol(training_dummied_fac_Miss)],
                     training_dummied_fac_Miss$Comp_30,
                     sizes = varSeq,
                     method = 'avNNet',
                     preProc = c("center", "scale"),
                     tuneLength = 4,
                     trace = FALSE, 
                     lineout = TRUE,
                     metric = 'ROC',
                     trControl = trCtrl,
                     rfeControl = rfeCtrl)


cl <- makeCluster(15, outfile = '')
registerDoParallel(cl) # Change to acceptable number of cores based on your feasability
getDoParWorkers()
stopCluster(cl) # Stop cluster computations
registerDoSEQ() # Unregister doParallel
