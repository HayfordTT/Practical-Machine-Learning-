---
title: "Practical Machine Learning"
author: "Hayford Tetteh"
date: "2 June 2020"
output:
  word_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]

The test data are available here: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv]

The data for this project come from this source: [http://groupware.les.inf.puc-rio.br/har]. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details.


## Reproduceability

Our outcome variable is classe, a factor variable. For this data set, “participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions: - exactly according to the specification 
#### (Class A) - throwing the elbows to the front #### (Class B) - lifting the dumbbell only halfway #### (Class C) - lowering the dumbbell only halfway #### (Class D) - throwing the hips to the front (Class E)

Two models will be tested using decision tree and random forest. The model with the highest accuracy will be chosen as our final model.

## Cross-validation
Cross-validation will be performed by subsampling our training data set randomly without replacement into 2 subsamples: subTraining data (75% of the original Training data set) and subTesting data (25%). Our models will be fitted on the subTraining data set, and tested on the subTesting data. Once the most accurate model is choosen, it will be tested on the original Testing data set.

## Expected out-of-sample error
The expected out-of-sample error will correspond to the quantity: 1-accuracy in the cross-validation data. Accuracy is the proportion of correct classified observation over the total sample in the TestTrainingSet data set. Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error will correspond to the expected number of missclassified observations/total observations in the Test data set, which is the quantity: 1-accuracy found from the cross-validation data set.

Our outcome variable “classe” is a factor variable. We split the Training dataset into TrainTrainingSet and TestTrainingSet datasets.

#### importing necessary libraries
```{r libraries}
library(lattice); library(ggplot2); library(caret); library(randomForest); library(rpart); library(rpart.plot);
library(rattle); library(RColorBrewer)
library(shiny); library(knitr); library(knitLatex)
```


## Downloading The Data

```{r The Data}
### The training data set can be found on the following URL:
train_Data <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
### The testing data set can be found on the following URL:

test_Data <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```
### Loading the Data
```{r Loading the Data}
training <- read.csv(url(train_Data), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(test_Data), na.strings=c("NA","#DIV/0!",""))
```
### Viewing Head()
```{r Viewing Head()}
dim(training)
dim(testing)
```

### Delete columns with all missing values
```{r Delete columns}
training <-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]
```
### Some variables are irrelevant to our current project.
```{r Some variables}
training   <-training[,-c(1:7)]
testing <-testing[,-c(1:7)]
```

### looking at the new datasets:
```{r new datasets}
dim(training)
dim(testing)
```

## Partitioning the training data set to allow cross-validation

The training data set contains 53 variables and 19622 obs.
The testing data set contains 53 variables and 20 obs.
In order to perform cross-validation, the training data set is partionned into 2 sets: subTraining (75%) and subTest (25%).
This will be performed using random subsampling without replacement.

```{r Partitioning the training}
# partition the data so that 75% of the training dataset into training and the remaining 25% to testing
sub_train <- createDataPartition(y=training$class, p=0.75, list=FALSE)
Sub_Training <- training[sub_train, ] 
Sub_Testing <- training[-sub_train, ]
```
#### Look at the dim()
```{r the dim()}
dim(Sub_Training)
dim(Sub_Testing)
```

### A look at the Data

The variable “classe” contains 5 levels: A, B, C, D and E. A plot of the outcome variable will allow us to see the frequency of each levels in the subTraining data set and compare one another.
```{r look at the Data}
plot(Sub_Training$class, col="navy", main="Variable classes within the Sub_Training data set", xlab="Classes levels", ylab="Frequency")
```

## Using ML algorithms for prediction: Decision Tree
```{r ML algorithms}
modFitA1 <- rpart(classe ~ ., data=Sub_Training, method="class")
```

### To view the decision tree with fancy :
```{r the decision}
fancyRpartPlot(modFitA1)
```

```{r modFitA1}
modFitA1 <- rpart(classe ~ ., data=Sub_Training, method="class")

# Predicting:
prediction1 <- predict(modFitA1, Sub_Testing, type = "class")

# Plot of the Decision Tree
rpart.plot(modFitA1, main="Classification Tree", extra=102, under=TRUE, faclen=0)
```

## My Predictions:

```{r My Predictions:}
predictions1 <- predict(modFitA1, Sub_Testing, type = "class")
```

### Using confusion Matrix to test results:
```{r Using confusion1}
confusionMatrix(predictions1, Sub_Testing$classe)
```

## Using ML algorithms for prediction: Random Forests
```{r Using ML algorithms}
modFitA2 <- randomForest(classe ~., data = Sub_Training)
```

### Predicting in-sample error:
```{r Predicting}
predictionsA2 <- predict(modFitA2, Sub_Testing, type = "class")
```
### Using confusion Matrix to test results:
```{r Using confusion}
confusionMatrix(predictionsA2, Sub_Testing$classe)
```

## Decision

As expected, Random Forest algorithm performed better than Decision Trees.
Accuracy for Random Forest model was 0.995 (95% CI: (0.993, 0.997)) compared to 0.739 (95% CI: (0.693, 0.719)) for Decision Tree model. The random Forest model is choosen. The accuracy of the model is 0.995. The expected out-of-sample error is estimated at 0.005, or 0.5%. The expected out-of-sample error is calculated as 1 - accuracy for predictions made against the cross-validation set. Our Test data set comprises 20 cases. With an accuracy above 99% on our cross-validation data, we can expect that very few, or none, of the test samples will be missclassified.

## Submission
```{r Submission}
# predict outcome levels on the original Testing data set using Random Forest algorithm
predictfinal <- predict(modFitA2, testing, type="class")
predictfinal
```

## predict outcome levels on the original Testing data set using Random Forest algorithm
```{r predict outcome}
predictfinal <- predict(modFitA1, testing, type="class")
predictfinal
```

## Write files for submission
```{r files for submission}
machine_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

machine_write_files(predictfinal)
```