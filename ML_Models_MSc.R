# Run data cleaning script
source('Data_Cleaning_MSc.R')


# Loading required packages
library(caret)
library(doParallel) # Parallel computations


# Parallelizing computations
makeCluster(detectCores())
registerDoParallel(15) # Change to acceptable number of cores based on your feasability
getDoParWorkers() 
stopCluster(cl) # Stop cluster computations



# Basic Models ----------------------------------------------------------------------



