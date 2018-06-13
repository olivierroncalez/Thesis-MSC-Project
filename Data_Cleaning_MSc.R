# Data Loading ----------------------------------------------------------------------
# This script is automatically run from "ML_Models_MSc.R" in order to keep the data 
# cleaning and model building separate.  

library(tidyverse)
library(clipr)
library(devtools)
library(DataExplorer)
library(data.table)
library(lubridate)
library(outliers)
library(magrittr)





# Setting working directory
setwd("~/Desktop/R Code/Thesis Project")

# Reading the data in tibble 
data <- read_csv("complications-ds-proj-18.csv")





# Data Summary ----------------------------------------------------------------------


###########################################
# Exploration - Overview
###########################################


# Glimpsing at the data
glimpse(data)

# Summary
summary(data)

## Group by analyses
# Number of people in each hospital
data %>% group_by(Group) %>% summarise(n = n())



###########################################
# Data Types
###########################################


# Variable types (as dataframe)
variable_types <- as.data.frame(sapply(data, class)) %>% 
  rownames_to_column() %>% 
  setnames(c('Name', 'Class'))


### Explorations for variable types

# Character
sum(sapply(data, is.character)) # Number of character types
names(data[,sapply(data, is.character)]) # Names of string columns 

# Numeric
sum(sapply(data, is.numeric)) # Nuuni_condmber of integer columns
names(data[,sapply(data, is.numeric)]) # Names of integer columns


# Dimensions of the data
dim(data)
nrow(data)
ncol(data) 


# Removing old env values
rm('variable_types') # Comment out if you wish to keep this. Removed as this is exploratory.





# Missing Data ----------------------------------------------------------------------


# Number of variables with NA globally
sum(is.na(data))


# Number of variables with NA by variable
missing_values <- apply(data, 2, function(x) sum(is.na(x))) %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  setnames(c('Name', 'Missing_Count'))


# Arange missing values by descending order of values
arrange(missing_values, desc(Missing_Count))


# Removing old env values
rm('missing_values') # Comment out if you wish to keep this. Removed as this is exploratory.





# Frequencies  ----------------------------------------------------------------------


###########################################
# Table Count
###########################################


# === Apply table function to each variable in the data if the length is less than 11. 
table_count <- data %>% 
  lapply(function(x) if(length(table(x)) < 14) table(x))


# === Remove all list elements that are null
table_count <- table_count[!sapply(table_count, is.null)] %>% 
  lapply(as.data.frame) %>% # save each list elmt as df
  lapply(spread, x, Freq) # spread the rows into columns


# Print tables
table_count


# === Names of variables in the list 
names(table_count)


# Removing old env values
rm('table_count') # Comment out if you wish to keep this. Removed as this is exploratory.





###########################################
# Unique Count
###########################################

# Conditions
# === Extract the names of variables who's table length is greater than 14 
# === In other words, those who were not made into the table above
tbl_len <- data %>%  lapply(function (x) length(table(x))) %>% unlist %>% as.vector # len of table count
uniq_cond <- names(data[tbl_len > 14]) # Creating subset condition


# Unique Counts
unique_count <- data %>% 
  select(uniq_cond) %>% # Select all variables according to code above
  select(-contains('date'), -id, -Age, - Pred) %>% # Remove certain variables (dates, id, etc...)
  lapply(function(x) unique(x)) # Get unqiue values


# Print unique counts
unique_count


# Removing old env values
rm('tbl_len', 'uniq_cond', 'unique_count') # Comment out if you wish to keep this. 
                                           # Removed as this is exploratory. 
                                           # tbl_len & uniq_cond removed regardless.




# Orange ----------------------------------------------------------------------------


# Selecting Orange Vars
orange <- select(data, `*PreviousCardiov10cularDise10e`, `*PreviousRespiratoryDisease`, 
                 `*PreviousAbdominalDisease`, Other, `*PreviousOtherDisease`, PreviousTreatment, 
                 T, starts_with("Proc"), Fluid_2, starts_with("Local"), starts_with("General"), 
                 Severity, Days_30)


# Examining unique values for all Orange vars
orange %>%  lapply(function(x) unique(x))


# Removing old env values
rm('orange') # Comment out if you wish to keep this. Removed as this is exploratory.





# CVS, RESP, Other Concordance ----------------------------------------------------------------------


## Respiratory

# Resp (General)
filter(data, RESP == 0) %>% 
     nrow()
filter(data, RESP == 1) %>% 
     nrow()
filter(data, is.na(RESP)) %>% # 9 missing values
     nrow()


# RESP == 0
filter(data, RESP == 0, `*PreviousRespiratoryDisease` == 0) %>% 
     nrow()
filter(data, RESP == 0, `*PreviousRespiratoryDisease` != 0) %>% 
     nrow() # 5 values are misattributed 
filter(data, RESP == 0, is.na(`*PreviousRespiratoryDisease`)) %>% 
     nrow() # 11 missing


# RESP == 1
filter(data, RESP == 1, `*PreviousRespiratoryDisease` == 0) %>% 
     nrow() # 1 value misattributed
filter(data, RESP == 1, `*PreviousRespiratoryDisease` != 0) %>% 
     nrow() 
filter(data, RESP == 1, is.na(`*PreviousRespiratoryDisease`)) %>% 
     nrow()


# Examination of misattributions (Resp)
filter(data, RESP == 0, `*PreviousRespiratoryDisease` != 0) %>% 
  select(contains('resp'))
filter(data, RESP == 1, `*PreviousRespiratoryDisease` == 0) %>% 
  select(contains('resp'))





## Abdominal Disease

# ABDO == 0
filter(data, ABDO == 0, `*PreviousAbdominalDisease` == 0) %>% 
     nrow()
filter(data, ABDO == 0, `*PreviousAbdominalDisease` != 0)%>% 
     nrow() # 2 misattributions
filter(data, ABDO == 0, is.na(`*PreviousAbdominalDisease`)) %>% 
     nrow() # 12 missing


# ABDO == 1
filter(data, ABDO == 1, `*PreviousAbdominalDisease` == 0) %>% 
     nrow()
filter(data, ABDO == 1, `*PreviousAbdominalDisease` != 0) %>% 
     nrow() 
filter(data, ABDO == 1, is.na(`*PreviousAbdominalDisease`)) %>% 
     nrow() 


# Examination of misattributions (ABDO)
filter(data, ABDO == 0, `*PreviousAbdominalDisease` != 0) %>% 
  select(contains('Abdo'))

 
                                                           
                                                            
                                                            
## Other 

# Other == 0
filter(data, Other == 0, `*PreviousOtherDisease` == 0) %>% 
     nrow()
filter(data, Other == 0, `*PreviousOtherDisease` != 0) %>% 
     nrow() # 2 misattributions
filter(data, Other == 0, is.na(`*PreviousOtherDisease`)) %>% 
     nrow() # 5 missing


# Other == 1
filter(data, Other == 1, `*PreviousOtherDisease` == 0) %>% 
     nrow()
filter(data, Other == 1, `*PreviousOtherDisease` != 0) %>% 
     nrow() 
filter(data, Other == 1, is.na(`*PreviousOtherDisease`)) %>% 
     nrow() 


# Examination of misattributions (Other)
filter(data, Other == 0, `*PreviousOtherDisease` != 0) %>% 
  select(contains('Other'))





###########################################
# Group Missingness
###########################################


# === Retrieve proportion of missing values by variable per group
proportion_missing <- data %>% 
  group_by(Group) %>% 
  summarise_all(funs(sum(is.na(.))/length(.)))


# === If group does not have any values for a variable, 1, else, 0
proportion_missing <- sapply(proportion_missing, function(x) ifelse(x == 1, 1, 0)) %>% 
  as.data.frame
# === Selecting only variables with at least one '1'
proportion_missing <- select_if(proportion_missing, function(x) sum(x) > 0)
# === Removing the variable group
proportion_missing <- select(proportion_missing, -Group, -duplicate)




## Variables completely missing by group

# === Getting the names of all variables which 
group_missing_var_names <- proportion_missing %>% apply(1, function(x) names(which(x == 1)))
group_missing_var_names # Printing results


# Examine difference in missing variables between groups
setdiff(group_missing_var_names[[4]], group_missing_var_names[[5]])


# Removing old env values
rm('group_missing_var_names', 'proportion_missing')



# Data Cleaning ---------------------------------------------------------------------


###########################################
# Removing Variables
###########################################


# === See data dictionary for explanations for removal

# Variable removal (Isabel recommendations)
data <- select(data, 
               -MUSTNutritionScore, # Deemed unreliable
               -BloodLoss, # Inconsistent (Isabel)
               -Daysdeath, # Previously derived
               -Pred, # Can ignore (Isabel)
               -PreviousTreatment) # Not consistent/String


# 'Orange variables' - i.e., Inconsistent variables
data <- select(data,
               -`*PreviousCardiov10cularDise10e`,
               -`*PreviousRespiratoryDisease`,
               -`*PreviousAbdominalDisease`,
               -Other,
               -`*PreviousOtherDisease`,
               -starts_with("Proc"), 
               -starts_with("Local"), 
               -starts_with("General"))





###########################################
# Removing Rows
###########################################


# Removing single duplicate row
data <- filter(data, is.na(duplicate))


# Remove variable duplicate 
#=== Uninformative as all values are now 0.
data <- select(data, -duplicate)





###########################################
# Dates and Times Manipulation
###########################################


# Date time conversion (lubridate)
data$Dateofop <- dmy(data$Dateofop)
data$DischargeDate <- dmy(data$DischargeDate) 
data$DateofDeath <- dmy(data$DateofDeath) # 36 parsing errors (concerns dates "01/01/1900")
                                          # Changed to NA





###########################################
# Outlier & eronneous values detection - Part I
###########################################


# === Note: This section is closely tied with the variable manipulation section
# === below. The results of the observations here guided some of the manipulations
# === Additional note, there are several parts to these 2 topics which is done in 
# === order to allow the code to be executed sequentially. In other words, some 
# === cleaning may need to be done before continued exploration of outliers or eroneous values.


## Examining 'character' data

# Name extraction
names(select_if(data, is.character)) # Obtaining the names of character values
# Table examination
table(data$Severity)
table(data$`Clavien-Dindo`)
table(data$T)
table(data$Days_30)





###########################################
# Other Variable Manipulation - Part I
###########################################


## Standardization

# Standardizing severity/CD scores letter
data$Severity <- str_to_upper(data$Severity)
data$`Clavien-Dindo` <- str_to_upper(data$`Clavien-Dindo`)
table(data$Severity) # View results
table(data$`Clavien-Dindo`) # View results




## Conversion of Values

# === Changing 1 instance of x to 0. Conversion to numeric
data$T <- str_replace(data$T, 'x', 'NA') %>% as.numeric
table(data$T) # View results


# === Changing patient hospital transfer as missing data (1 instance). Conversion to numeric.
data$Days_30 <- str_replace(data$Days_30, 'transferred to RSCH', 'NA') %>% as.numeric
table(data$Days_30) # View results (potential near zero var)


# === Not sure what Fluid_2 is, but I've discretized it to >= 1, and 0. Only 4 values were
# === greater than 1 (1.5, 2c, 3, 4; 1 value for each). Transformed to numeric.
data$Fluid_2 <- data$Fluid_2 %>% 
  str_replace('2c', '2') %>%
  as.numeric %>%
  sapply(function(x) ifelse(x > 1, 1, x))
table(data$Fluid_2)


# Group changes
# === As there is no 'Group 3', groups 4-6 have been bumped down a level.
# === Note: Running this more than once without resetting the data will cause group labeling errors.
data$Group[data$Group == 4] <- 3
data$Group[data$Group == 5] <- 4
data$Group[data$Group == 6] <- 5





###########################################
# More concordance checks
###########################################


## Concordance Severity & Clavien-Dindo scores

# === Checking concordence between both scores and if one is missing but the other is present. 

# === Is one missing but the other present?
data %>% select(Severity, `Clavien-Dindo`) %>% 
  filter(!is.na(Severity) & is.na(`Clavien-Dindo`) | is.na(Severity) & !is.na(`Clavien-Dindo`))


# === Are the two scores present but different?
data %>% 
  select(Severity, `Clavien-Dindo`) %>% 
  filter(Severity != `Clavien-Dindo`) # For 207 rows concordance does not hold. 




## Concordance post-op complications

# Complications and Clavien-Dindo + Severity
# === Creating new dataset for easier manipulatuion
post_ops <- data %>% 
  select(Complications, Severity, `Clavien-Dindo`) %>% 
  rename_all(funs(c('C', 'S', 'CD')))



# Number of rows where a data exists for (either CD or Severity) AND Complications = 635
post_ops %>% filter((!is.na(S) | !is.na(CD)) &
                      !is.na(C)) %>% 
  summarize(n = n())


# Some index of complication, but no complication data = 84
post_ops %>% filter((!is.na(S) | !is.na(CD)) & 
                      is.na(C)) %>% 
  summarize(n = n())


# No index of complication, but complication marked = 9
post_ops %>% filter((is.na(S) & is.na(CD)) & 
                      !is.na(C)) %>% 
  summarize(n = n())



# No complication, but index diff than 0 (either one)
post_ops %>% filter((S != 0 | CD != 0) & # != 0 implicitly removes NAs. 
                      C == 0) %>% 
  summarize(n = n())


# Complication, but index == 0 (either one if not missing) = 12
post_ops %>% filter(
     ((S == 0 & CD == 0) | (S == 0 & is.na(CD)) | (is.na(S) & CD == 0)) & 
    C != 0) %>% 
  summarize(n = n())

# No complication, but index != - (either one if not missing) = 1
post_ops %>% filter(
     ((S != 0 & CD != 0) | (S != 0 & is.na(CD)) | (is.na(S) & CD != 0)) & 
          C == 0) %>% 
     summarize(n = n())

# Removing temporary data
rm('post_ops')




## Concordance anesthesia

# Boolean greater than 6 hours, numeric greater than 6 = 161 instances (concordance)
data %>% select(ANAESTHETIC__6hrs, AnaestheticTime_hours_) %>% 
     filter(AnaestheticTime_hours_ >= 6 & ANAESTHETIC__6hrs == 1) %>% 
     tally()


# Boolean less than 6 hours, numeric less than 6 = 315 instances (concordance)
data %>% select(ANAESTHETIC__6hrs, AnaestheticTime_hours_) %>% 
     filter(AnaestheticTime_hours_ < 6 & ANAESTHETIC__6hrs == 0) %>% 
     tally()


# Boolean NOT greater than 6 hours, numeric greater than 6 = 14 instances (no concordance)
data %>% select(ANAESTHETIC__6hrs, AnaestheticTime_hours_) %>% 
     filter(AnaestheticTime_hours_ >= 6 & ANAESTHETIC__6hrs != 1) %>% 
     tally()


# Boolean greater than 6 hours, numeric NOT greater than 6 = 1 instance (no concordance)
data %>% select(ANAESTHETIC__6hrs, AnaestheticTime_hours_) %>% 
     filter(AnaestheticTime_hours_ < 6 & ANAESTHETIC__6hrs != 0) %>% 
     tally()

# Group_by exploration of concordance
data %>% select(ANAESTHETIC__6hrs, AnaestheticTime_hours_) %>% 
     group_by(ANAESTHETIC__6hrs != 0, AnaestheticTime_hours_ >= 6) %>% 
     tally %>% 
     na.omit # Remove any rows with missing values





## Concordance complications & complications number 

# Complications 'yes', complications number != 0 (concordance)
data %>% select(Complications, ComplicationNumber) %>% 
     filter(Complications == 1, ComplicationNumber != 0) %>% 
     tally()

# Complications 'no', complications number == 0 (concordance)
data %>% select(Complications, ComplicationNumber) %>% 
     filter(Complications == 0, ComplicationNumber == 0) %>% 
     tally()

# Complications 'yes', complications number == 0 (no concordance)
data %>% select(Complications, ComplicationNumber) %>% 
     filter(Complications != 0, ComplicationNumber == 0) %>% 
     tally()

# Complications 'no', complications number != 0 (no concordance)
data %>% select(Complications, ComplicationNumber) %>% 
     filter(Complications == 0, ComplicationNumber != 0) %>% 
     tally()

# Group by exploration of concordance
data %>% select(Complications, ComplicationNumber) %>% 
     group_by(Complications, ComplicationNumber == 0) %>%  
     tally %>% 
     na.omit




## Concordance died & days_30

# Boolean did not die (concordance)
data %>% select(Died, Days_30) %>% 
     filter(Died == 0, Days_30 == 0) %>% 
     tally()

# Boolean died (concordance)
data %>% select(Died, Days_30) %>% 
     filter(Died == 1, Days_30 == 1) %>% 
     tally()

# Boolean did not die, boolean died days_30 (no concordance)
data %>% select(Died, Days_30) %>% 
     filter(Died == 0, Days_30 == 1) %>% 
     tally()

# Boolean died, boolean did not die days_30 
# === This does NOT imply lack of concordance. Patient may die before 30 days. 
data %>% select(Died, Days_30) %>% 
     filter(Died == 1, Days_30 == 0) %>% 
     tally()

# Group_by exploration of concordance
data %>% select(Died, Days_30) %>% 
     group_by(Died, Days_30) %>% 
     tally %>% 
     na.omit




## Concordance hospital stay

# Days numeric greater or equal to 35, boolean greater than 35 (concordance)
data %>% select(Daysinpatient, Daysinpatient_35) %>% 
     filter(Daysinpatient >= 35, Daysinpatient_35 == 1) %>% 
     tally()

# Days numeric less than 35, boolean less than 35 (concordance)
data %>% select(Daysinpatient, Daysinpatient_35) %>% 
     filter(Daysinpatient < 35, Daysinpatient_35 == 0) %>% 
     tally()


# Days numeric less than 35, boolean greater than 35 (no concordance)
data %>% select(Daysinpatient, Daysinpatient_35) %>% 
     filter(Daysinpatient < 35, Daysinpatient_35 != 0) %>% 
     tally()

# Days numeric greater or equal to 35, boolean less than 35 (no concordance)
data %>% select(Daysinpatient, Daysinpatient_35) %>% 
     filter(Daysinpatient >= 35, Daysinpatient_35 == 0) %>% 
     tally()

# Group_by exploration of concordance
data %>% select(Daysinpatient, Daysinpatient_35) %>% 
     group_by(Daysinpatient >= 35, Daysinpatient_35 == 0) %>% 
     tally %>% 
     na.omit




###########################################
# Outlier & eronneous values detection - Part II
###########################################


# Examining Range
ranges <- data %>% 
  select_if(is.numeric) %>% # Grab numeric variables
  sapply(range, na.rm = T) %>%  # Apply range function
  as.data.frame %>% # Convert to df
  rownames_to_column %>% # Add row names as column
  gather(Variable, value, -rowname) %>% # Gather key-value pairs (except rownames)
  spread(rowname, value) %>% # Create columns as rownames, spreading the values
  rename_all(funs(c('Variable', 'Min', 'Max')))

# === ranges value will be removed and replaced from the environment below. It is only used
# === for data preprocessing exploration (identification of problematic values)




## Examining table counts of categorical variables

# === Note: need to verify these variables are indeed categorical & whether ordinal or nominal.
table_count <- data %>% select_if(is.numeric) %>% # Selects numeric variables
  sapply(function(x) if(length(table(x)) < 15) table(x)) # Applies table if length categories < 15


# === Filters null values and transforms the list into dataframes
table_count <- table_count[!sapply(table_count, is.null)] %>% # Remove null list values
  lapply(as.data.frame) %>% 
  lapply(spread, x, Freq) # Spread keys into variables


# === Save var names & Bind rows into easy-to-view-df
categorical_names <- names(table_count) # Temporarily save the names in the list
table_count <- bind_rows(table_count) # Bind all the list rows together into single df


# === Re-ordering the column names and inserting variable names as additional column.
table_count <- bind_cols(table_count, as.data.frame(categorical_names))
table_count <- table_count[, c('categorical_names', 0:6)] %>% 
  rename(Variable = categorical_names)

# === Note that these 'categorical' variables may not be categorical and vice versa. Need
# === to check that variables have been encoded properly using domain knoweldge rather than 
# === just using the heuristic of unique value lengths smaller than 15. 





###########################################
# Other Variable Manipulation - Part II
###########################################

## Conversion of values

# Out of permissable range 
# === Values outside of normal ranges are transformed into NAs. 
# === Negative values were converted to missing
data$Daysinpatient <- data$Daysinpatient %>% 
  sapply(function(x) ifelse(x < 0, NA, x))






###########################################
# Finalizing variable selection - Data Cleaning
###########################################

# === Based on discussions with my supervisor, the following variables are removed.
# === Reference 06/06/18 discusssion and variable list for more detail.

# Removing remaining variables
data <- data %>% select(-c(id, Dateofop, ACE_27, ASA, AnaestheticTime_hours_, Fluid_2, NonbloodinfusedL,
                           Complications, ComplicationNumber, Severity, DischargeDate, Died, Days_30, DateofDeath,
                           Daysinpatient, Daysinpatient_35, `Clavien-Dindo`, `CD>3`, Wound, Cardiac, Pulmonary,
                           Flap_failure))


# Removing variables with more than 50% missing data
data <- data[-which(rowMeans(is.na(data)) > .5),] 


# Removing cases with no labelled outcome
data <- filter(data, !is.na(Comp_30))




###########################################
# Factor conversion
###########################################

# Data converted to factors here
fac_names <- data %>%  select(-c(Age, N)) %>% names # Extract names of soon-to-be factor variables.
# Note: Domain knowledge guided this process here.
data %<>% mutate_at(fac_names, funs(factor(.))) # Update data by mutating to factor variables


# Factor names (verifying)
data %>% sapply(function(x) is.factor(x)) %>% .[. %in% TRUE] %>% names
data %>% sapply(function(x) is.numeric(x)) %>% unname


# Labelling class data
data$Comp_30 <- factor(data$Comp_30, labels = c('Healthy', 'Comp'))


# Moved new data set creation below.





# EDA (Clean Data) --------------------------------------------------------


# === Note that this mainly re-runs the same exploratory functions as above on the 
# === cleaned data.

# Remove old env values
rm('ranges', 'table_count', 'categorical_names')





###########################################
# Exploration - Overview
###########################################


# Glimpsing at the data
glimpse(data)
head(data, n = 10)


# Summary
summary(data)


## Group by analyses
data %>% group_by(Group) %>% tally





###########################################
# Data Types
###########################################


# Variable types (as dataframe)
variable_types_f <- as.data.frame(sapply(data, class)) %>% # f for 'final'
  rownames_to_column() %>% 
  setnames(c('Name', 'Class'))




## Explorations for variable types

# Character
sum(sapply(data, is.character)) # Number of character types
names(data[,sapply(data, is.character)]) # Names of string columns 

# Numeric
sum(sapply(data, is.numeric)) # Number of integer columns
names(data[,sapply(data, is.numeric)]) # Names of integer columns

# Factors
sum(sapply(data, is.factor))
names(data[, sapply(data, is.factor)])





###########################################
# Missing Data
###########################################

## Missing values by variable 

# Number of variables with NA globally
sum(is.na(data))


# Number of variables with NA by variable
missing_values_f <- apply(data, 2, function(x) sum(is.na(x))) %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  setnames(c('Name', 'Missing_Count'))


# Arange missing values by descending order of values
arrange(missing_values_f, desc(Missing_Count))


# Visualizing missing values
ggplot(data = missing_values_f, aes(x = reorder(Name, -Missing_Count), y = Missing_Count)) +
     geom_bar(stat = 'identity', fill = 'steelblue') +
     theme_minimal() +
     theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
     labs(title = 'Variabel Missing Values in Decreasing Order') +
     scale_x_discrete(name = 'Variable') +
     scale_y_continuous(name = 'Frequency of Missing Values', breaks = seq(10, 250, 10))



## Missingness by group

# === Retrieve proportion of missing values by variable per group
proportion_missing_f <- data %>% 
  group_by(Group) %>% 
  summarise_all(funs(sum(is.na(.))/length(.)))


# === If group does not have any values for a variable, 1, else, 0
proportion_missing_f <- sapply(proportion_missing_f, function(x) ifelse(x == 1, 1, 0)) %>% 
     as.data.frame
# === Selecting only variables with at least one '1'
proportion_missing_f <- select_if(proportion_missing_f, function(x) sum(x) > 0)
# === Removing the variable group
proportion_missing_f <- select(proportion_missing_f, -Group)




## Variables completely missing by group

# === Getting the names of all variables which 
group_missing_var_names_f <- proportion_missing_f %>% apply(1, function(x) names(which(x == 1)))
group_missing_var_names_f # Printing results


# Examine difference in missing variables between groups
setdiff(group_missing_var_names_f[[4]], group_missing_var_names_f[[5]])




### Patterns of missing data

library(VIM)
aggr_plot <- aggr(data, 
                  col = c('navyblue','red'), 
                  numbers = TRUE, 
                  sortVars = TRUE, 
                  labels = names(data), 
                  cex.axis = .7, 
                  gap = 3, 
                  ylab = c("Histogram of missing data","Pattern"))



###########################################
# Table Count 
###########################################


# === This is the same as the analysis previous run. Nothing will change unless other
# === data cleaning has occured. Note that this does not include `Clavien-Dindo` or 
# === Severity scores (which are considered as strings due to letters accompanying the scores).
# === Domain knowledge needed to verify heuristics of categorical-level data. 




## Table count DataFrame

# === Note: need to verify these variables are indeed categorical & whether ordinal or nominal.
table_count_f <- data %>% select_if(is.factor) %>% # Selects factor variables
  sapply(function(x) if(length(table(x)) < 16) table(x)) # Applies table if length categories < 16


# === Filters null values and transforms the list into dataframes
table_count_f <- table_count_f[!sapply(table_count_f, is.null)] %>% # Remove null list values
  lapply(as.data.frame) %>% 
  lapply(spread, x, Freq) # Spread keys into variables


# === Save var names & Bind rows into easy-to-view-df
categorical_names <- names(table_count_f) # Temporarily save the names in the list
table_count_f <- bind_rows(table_count_f) # Bind all the list rows together into single df


# === Re-ordering the column names and inserting variable names as additional column.
table_count_f <- bind_cols(table_count_f, as.data.frame(categorical_names))
table_count_f <- table_count_f[, c('categorical_names', 0:5)] %>% 
  rename(Variable = categorical_names)







###########################################
# Variable Range & Concordance
###########################################

# Examining Range
ranges_f <- data %>% 
  select_if(is.numeric) %>% # Grab numeric variables
  sapply(range, na.rm = T) %>%  # Apply range function
  as.data.frame %>% # Convert to df
  rownames_to_column %>% # Add row names as column
  gather(Variable, value, -rowname) %>% # Gather key-value pairs (except rownames)
  spread(rowname, value) %>% # Create columns as rownames, spreading the values
  rename_all(funs(c('Variable', 'Min', 'Max')))





# Visualizations --------------------------------------------------------------------



# To complete... 





# Datasets --------------------------------------------------------------------------

#
#
# In addition to the raw data, several modified copies of the data have been generated here in order
# to fit in a format acceptable for specific ML algorithms, as well as to determine the differences in 
# effects of different formats.
#
#




### Data with missing categorical data as factor level (MISSING DATA FACTOR LEVEL)


# New dataset which treats missing values as additional factors (for factor variables)
data_facMiss <- data %>% mutate_at(fac_names, funs(fct_explicit_na(.)))


# Verifying class and levels
data_facMiss %>% sapply(class) %>% unname # class
data_facMiss %>% lapply(function(x) levels(x)) # levels
data_facMiss %>% sapply(function(x) sum(is.na(x))) %>% .[. > 0] # Columns with remaining missing values






### Dataset with listwise deletion of missing data (LISTWISE DELETION)


# Data with listwise deletion of missing values
data_noMiss <- na.omit(data) # Not desirable


# Percent of data with at least one missing value
sum(complete.cases(data))/nrow(data) # 57% of data have no missing values (43% were thus eliminated)







# Dummy Variables -------------------------------------------------------------------


### Dummy variables for encoded missing data in factors 


# Create new data & dummy object 
data_facMiss_dummied <- data_facMiss %>% select(-Group) # New data (excluding group)
dummy_obj <- dummyVars(~. -Comp_30, data_facMiss_dummied, fullRank = TRUE) # Create dummy variable encoding object


# Replace dataset with new dummy variables
data_facMiss_dummied <- as.data.frame(predict(dummy_obj, newdata = data_facMiss_dummied)) %>% 
     bind_cols(., data_facMiss_dummied[ncol(data_facMiss_dummied)])


# Detect and eliminate (near) zero variance predictors
nearZeroVar(data_facMiss_dummied, saveMetrics = TRUE) %>% rownames_to_column(var = 'Variable') %>% 
     filter(nzv == TRUE) # Printout of variables 
nzv <- nearZeroVar(data_facMiss_dummied, saveMetrics = FALSE) # Retrieve indices of near zero var predictors
data_facMiss_dummied %<>% select(-nzv) # Eliminate variables with near zero var
data_facMiss_dummied %<>% select(-c(`Margins.(Missing)`, `Extracaps.(Missing)`)) # Eliminated two additional variables which proved to be
# near zero variance in the training set.



data_facMiss_dummied %>% sapply(function(x) sum(is.na(x))) %>% .[. > 0]











###########################################
# Clean up
###########################################

# Removing extraneous environment objects
rm(list = setdiff(ls(), c('data', 'data_facMiss', 'data_noMiss', 'data_facMiss_dummied')))
cat("\014") # Clear console
dev.off() # Clear plots








