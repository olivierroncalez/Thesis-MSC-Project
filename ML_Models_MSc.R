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

# Holding 1/3 of the data as independent test set. Balanced partitioning
set.seed(337)
inTraining <- createDataPartition(data$Comp_30, p = 0.33, list = FALSE)

training <- data[inTraining, ]
testing <- data[-inTraining, ]

# Create reproducible folds
set.seed(337)
createMultiFolds(y = data$Comp_30, k = 10, times = 5)


# Control objects -------------------------------------------------------------------



