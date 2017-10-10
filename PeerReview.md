Overview
--------

Using devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit*, it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify *how much* of a particular activity they do, but they rarely quantify *how well they do it*. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of six participants. They were asked to perform barbell lifts correctly and incorrectly in five different ways. More information is available from the website [here](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) (see the section on the **Weight Lifting Exercise Dataset**).

After narrowing the training set down to 32 predictors, a model was fit using the random forest method---resulting in an expected accuracy of 99.8 percent.

Download Training and Testing Data
----------------------------------

Check for the files in your working directory. If they do not already exist, download the `pml-training.csv` and `pml-testing.csv` files. Use `read.csv` to read in the data.

``` r
## Training set download
destfile_train <- "pml-training.csv"
fileURL_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

if(!file.exists(destfile_train)) {
        download.file(fileURL_train, destfile_train, method = "curl")
}

training <- read.csv(destfile_train, na.strings = c("NA", "#DIV/0!", ""))

## Testing set download
destfile_test <- "pml-testing.csv"
fileURL_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists(destfile_test)) {
        download.file(fileURL_test, destfile_test, method = "curl")
}

testing <- read.csv(destfile_test, na.strings = c("NA", "#DIV/0!", ""))
```

Preprocessing
-------------

First, load the {caret} package. For all preprocessing and model fitting, we are going to use the training data set. As a first step, check for variables showing little variance using the `nearZeroVar` function and remove those from the set.

``` r
library(caret)
nzv <- nearZeroVar(training)
preprocTraining <- training[,-nzv]
```

Remove the first five variables (X, user\_name, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2, cvtd\_timestamp) from the training set, as they should be irrelevant in predicting the classe variable.

``` r
preprocTraining <- preprocTraining[,-c(1:5)]
```

Using the `colSums` and `is.na` functions, find variables that contain missing information, and remove them from the training set.

``` r
preprocTraining <- preprocTraining[colSums(is.na(preprocTraining)) / nrow(preprocTraining) < .01]
```

Using the `cor` and `findCorrelation` functions, find variables with at least 75 percent correlation, and remove them from the training set.

``` r
corVars <- cor(preprocTraining[,-54])
highCor <- findCorrelation(corVars, cutoff = .75)
preprocTraining <- preprocTraining[,-highCor]
```

Following each of these steps leaves us with 32 variables to predict classe.

``` r
dim(preprocTraining)
```

    ## [1] 19622    33

Exploratory Data Analysis
-------------------------

The following code creates a density plot, showing the top four influential variables from our final model (see **Variable Importance** below). As you will see, we can already visualize some of the separation among each class within certain predictors.

*(Note: The code for editing the appearance of the plot is borrowed from [here](https://topepo.github.io/caret/visualizations.html).)*

``` r
featurePlot(x = preprocTraining[,c(1,2,21,23)], 
            y = preprocTraining$classe, 
            plot = "density",
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5, 
            pch = "|", 
            layout = c(4, 1), 
            auto.key = list(columns = 5))
```

![](PeerReview_files/figure-markdown_github/plotting.variables-1.png)

Partitioning Data
-----------------

In order to make the research reproducible, set the seed. Then, use the `createDataPartition` function to separate the training set into 60 percent training data and 40 percent testing data. By building the model using the training set, and by applying it to the testing set, we are using the cross-validation technique to assess how the results of this analysis will generalize to an independent data set.

``` r
set.seed(2017-10-09)
inTrain <- createDataPartition(preprocTraining$classe, p = 0.6, list = FALSE)
cleanTraining <- preprocTraining[inTrain,]
cleanTesting <- preprocTraining[-inTrain,]
```

Model Fitting
-------------

I chose to begin with the random forest method of fitting because a) it is usually one of the [top-performing](http://sux13.github.io/DataScienceSpCourseNotes/8_PREDMACHLEARN/Practical_Machine_Learning_Course_Notes.html#random-forest) algorithms for prediction and b) as an extension of classification trees, it's an ideal method for predicting class, or factor, variables. One drawback of the random forest method is the greater chance of over-fitting, particularly on noisy data sets. Hopefully, by carefully pre-processing the data, we avoid this issue.

First, use the `randomForest` function to build the model. Then, use the `predict` function to apply the model to the testing set. Finally, use the `confusionMatrix` function to determine the accuracy of the model.

As shown by the matrix below, the random forest method was a good first instinct. Its accuracy was 99.8 percent. While some may continue to test other models, for the purpose of brevity---and because any improvement would be negligble---I have decided to stick with this model.

``` r
library(randomForest)
modFit <- randomForest(classe ~ ., data = cleanTraining)
pred <- predict(modFit, cleanTesting, type = "class")
confusionMatrix(pred, cleanTesting$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 2231    1    0    0    0
    ##          B    1 1515    2    0    1
    ##          C    0    2 1363    8    0
    ##          D    0    0    3 1276    2
    ##          E    0    0    0    2 1439
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9972          
    ##                  95% CI : (0.9958, 0.9982)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9965          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9996   0.9980   0.9963   0.9922   0.9979
    ## Specificity            0.9998   0.9994   0.9985   0.9992   0.9997
    ## Pos Pred Value         0.9996   0.9974   0.9927   0.9961   0.9986
    ## Neg Pred Value         0.9998   0.9995   0.9992   0.9985   0.9995
    ## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
    ## Detection Rate         0.2843   0.1931   0.1737   0.1626   0.1834
    ## Detection Prevalence   0.2845   0.1936   0.1750   0.1633   0.1837
    ## Balanced Accuracy      0.9997   0.9987   0.9974   0.9957   0.9988

### Expected Out-of-sample Error

To calculate the expected out-of-sample error, we simply take the accuracy from the matrix above and subtract it from 1. The result is 0.2 percent.

``` r
1 - confusionMatrix(pred, cleanTesting$classe)$overall[[1]]
```

    ## [1] 0.002803977

### Variable Importance

The `varImp` function allows us to see which predictors contributed most to the final model. The top five for this model were num\_window, yaw\_belt, magnet\_dumbbell\_z, pitch\_forearm, and magnet\_belt\_y.

``` r
library(dplyr)
v <- varImp(modFit) %>% 
        mutate(var = row.names(varImp(modFit))) %>% 
        arrange(desc(Overall)) %>% 
        select(var, Overall)

library(pander)
pandoc.table(head(v, 5), justify = "left")
```

<table style="width:40%;">
<colgroup>
<col width="27%" />
<col width="12%" />
</colgroup>
<thead>
<tr class="header">
<th align="left">var</th>
<th align="left">Overall</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">num_window</td>
<td align="left">1138</td>
</tr>
<tr class="even">
<td align="left">yaw_belt</td>
<td align="left">771.6</td>
</tr>
<tr class="odd">
<td align="left">magnet_dumbbell_z</td>
<td align="left">587.4</td>
</tr>
<tr class="even">
<td align="left">pitch_forearm</td>
<td align="left">573.9</td>
</tr>
<tr class="odd">
<td align="left">magnet_belt_y</td>
<td align="left">532.5</td>
</tr>
</tbody>
</table>

Predictions
-----------

To predict the outcome variable on the provided testing set, we again use the `predict` function. The results are below.

``` r
predTest <- predict(modFit, testing, type = "class")
finalPredictions <- data.frame(problem.id = testing$problem_id, pred.classe = predTest)
pandoc.table(finalPredictions, justify = "left")
```

<table style="width:36%;">
<colgroup>
<col width="18%" />
<col width="18%" />
</colgroup>
<thead>
<tr class="header">
<th align="left">problem.id</th>
<th align="left">pred.classe</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">1</td>
<td align="left">B</td>
</tr>
<tr class="even">
<td align="left">2</td>
<td align="left">A</td>
</tr>
<tr class="odd">
<td align="left">3</td>
<td align="left">B</td>
</tr>
<tr class="even">
<td align="left">4</td>
<td align="left">A</td>
</tr>
<tr class="odd">
<td align="left">5</td>
<td align="left">A</td>
</tr>
<tr class="even">
<td align="left">6</td>
<td align="left">E</td>
</tr>
<tr class="odd">
<td align="left">7</td>
<td align="left">D</td>
</tr>
<tr class="even">
<td align="left">8</td>
<td align="left">B</td>
</tr>
<tr class="odd">
<td align="left">9</td>
<td align="left">A</td>
</tr>
<tr class="even">
<td align="left">10</td>
<td align="left">A</td>
</tr>
<tr class="odd">
<td align="left">11</td>
<td align="left">B</td>
</tr>
<tr class="even">
<td align="left">12</td>
<td align="left">C</td>
</tr>
<tr class="odd">
<td align="left">13</td>
<td align="left">B</td>
</tr>
<tr class="even">
<td align="left">14</td>
<td align="left">A</td>
</tr>
<tr class="odd">
<td align="left">15</td>
<td align="left">E</td>
</tr>
<tr class="even">
<td align="left">16</td>
<td align="left">E</td>
</tr>
<tr class="odd">
<td align="left">17</td>
<td align="left">A</td>
</tr>
<tr class="even">
<td align="left">18</td>
<td align="left">B</td>
</tr>
<tr class="odd">
<td align="left">19</td>
<td align="left">B</td>
</tr>
<tr class="even">
<td align="left">20</td>
<td align="left">B</td>
</tr>
</tbody>
</table>
