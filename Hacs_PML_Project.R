# ==================================================
# Practical Machine Learning
# ---------------------------------------------------
#  Harold Cruz-Sanchez
#  September 24th 2015
# ==================================================

date()
sessionInfo()
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(knitr)
library(AppliedPredictiveModeling)
library(ggplot2)
library(reshape2)

#===================================================
# INPUT DATA
# Internet CSV File
# ---------------------------------------------------
#
# This dataset is licensed under the **Creative Commons license (CC BY-SA)**. 
# The CC BY-SA license means you can remix, tweak, and build upon this work 
# even for commercial purposes, as long as you credit the authors of the 
# original work and you license your new creations under the identical terms they are licensing to you.
#===================================================

train_data <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", na.strings = c('NA','#DIV/0!',''))

test_data <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", na.strings = c('NA','#DIV/0!',''))

dim(train_data); dim(test_data)  # data dimensions
colnames_train <- colnames(train_data)
colnames_test <- colnames(test_data)

ggplot(train_data, aes(x = classe)) + 
  geom_histogram()

#===================================================
# Cleaning data
# ---------------------------------------------------
# Removing variables with NA values
#===================================================
non_NAs <- function(x) {
  as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}

# Build vector of missing data or NA columns to drop in the Train set.
train_nonNA_count <- non_NAs(train_data)

train_drop_cols <- c()

for (cnt in 1:length(train_nonNA_count)) {
  if (train_nonNA_count[cnt] < nrow(train_data)) {
    train_drop_cols <- c(train_drop_cols, colnames_train[cnt])
  }
}

train_drop_cols # variables that may be eliminated

my_train_data <- train_data[,!(names(train_data) %in% train_drop_cols)]
dim(my_train_data)

# Build vector of missing data or NA columns to drop in Test set.
test_nonNA_count <- non_NAs(test_data)

test_drop_cols <- c() 

for (count in 1:length(test_nonNA_count)) {
  if (test_nonNA_count[count] < nrow(test_data)) {
    test_drop_cols <- c(test_drop_cols, colnames_test[count])
  }
}

test_drop_cols # variables that may be eliminated

my_test_data <- test_data[,!(names(test_data) %in% test_drop_cols)]
dim(my_test_data)

#===================================================
# Cleaning data
# ---------------------------------------------------
# Removing unnecessary variables
# Some variables were created during the pre-processing of the raw data,
# and have no interest in the present model: Colums 1 to 7
#===================================================

# Train set.

my_train_data <- my_train_data[,8:length(colnames(my_train_data))]
dim(my_train_data)

# Test set.

my_test_data <- my_test_data[,8:length(colnames(my_test_data))]
dim(my_test_data)

#===================================================
# Cleaning data
# ---------------------------------------------------
# Comparing variable number in Train and Test sets
#===================================================

all.equal(colnames(my_train_data),colnames(my_test_data))

mismatch <- c(colnames(my_train_data) == colnames(my_test_data)) 
mismatch

# Found that one variable (Column 53) is different
# Find what variables are involve

names(my_train_data)[53]; names(my_test_data)[53]
# [1] "classe"
# [1] "problem_id"

#===================================================
# Predicting Model
# ---------------------------------------------------
# Creating the Model
# Random Forest 
#-----------------------------------------------------
# The Train data set will be randomly splited into 2 data sets. 
# The first part (75% of all data) will be used as trainig set, 
# while the rest (25%) will be used as validation set.
#===================================================
set.seed(123)

RF_Training <- createDataPartition(my_train_data$classe, p = .75, list = FALSE)
RF_my_data_Training <- my_train_data[ RF_Training,]
RF_my_data_validating  <- my_train_data[-RF_Training,]

set.seed(123)
RF_Fit_01 <- train(RF_my_data_Training$classe ~ ., data = RF_my_data_Training,
                   method = "rf",
                   trControl = trainControl(method = "cv", number = 4),
                   verbose = FALSE)
RF_Fit_01

print(RF_Fit_01$finalModel, digits=3)

predictions <- predict(RF_Fit_01, newdata=RF_my_data_validating)
print(confusionMatrix(predictions, RF_my_data_validating$classe), digits=4)

#===================================================
# Predicting Model
# ---------------------------------------------------
# Applying the Model
# Random Forest 
#-----------------------------------------------------
# Using the Test data set.
#===================================================

set.seed(123)
predict(RF_Fit_01, newdata=my_test_data)
print(predict(RF_Fit_01, newdata=my_test_data))

#===================================================
# Submitting answers
# ---------------------------------------------------
# Using the Test data set.
#===================================================

answers<-as.character(predict(RF_Fit_01, newdata=my_test_data))

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)

# ==================================================
# Practical Machine Learning
# ---------------------------------------------------
#  Harold Cruz-Sanchez
#  September 24th 2015
# ---------------------------------------------------
# REFERENCE: Thanks a lot to the authors
# Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. 
# Qualitative Activity Recognition of Weight Lifting Exercises. 
# Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). 
# Stuttgart, Germany: ACM SIGCHI, 2013.
# ==================================================

