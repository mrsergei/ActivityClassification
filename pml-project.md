# Predictive Activity Classification of Weight Lifting Exercises
January 23, 2015  
### Overview   
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to classify weight lifting exersies in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset). 



###Data analysis and preprocessing   

Loading raw data:

```r
trnA <- read.table("../data/pml-training.csv", 
                   header = TRUE, 
                   sep = ",",
                   fill = TRUE, 
                   stringsAsFactors = FALSE)

tstA <- read.table("../data/pml-testing.csv", 
                   header = TRUE, 
                   sep = ",",
                   fill = TRUE, 
                   stringsAsFactors = FALSE)

trnA$classe <- as.factor(trnA$classe)
```

Training data set contains 19622 observations of 160 features.
Analysis of the raw data we decided to first 7 columns that contain row number, user name, raw time stamps, window indicator and window number


```r
trn <- trnA[, -(1:7)]
tst <- tstA[, -(1:7)]
```
Further analysis of the data reviews many fetures containing large quantities of NAs

```r
nas <- apply(trn[1:(ncol(trn)-1)], 2, function(x){mean(is.na(suppressWarnings(as.numeric(x))))})
```
![](pml-project_files/figure-html/plotNAs-1.png) 

As you can see form the plot, there are two classes of features - ones that contain no NAs(52) and others that almos all NAs(100). For teh prupse of builidng the predictiove model we will reduce our data set only to features that have no NAs across observations.


```r
trn <- trn[, !(nas > 0)]
tst <- tst[, !(nas > 0)]
```

#####Data transformations
Subsequent analysis of the remaning features across training set observations revield s significant number of features with greater than 1 absoute value of skewness:


```r
skewVals <- apply(trn[, -ncol(trn)], 2, skewness)
sort(skewVals[abs(skewVals)>1], decreasing = TRUE)
```

```
##  gyros_dumbbell_z   gyros_forearm_z   gyros_forearm_y  gyros_dumbbell_y 
##        135.953437        116.076194         51.323792         31.648274 
## magnet_dumbbell_x     magnet_belt_x        pitch_belt      magnet_arm_z 
##          1.694024          1.433152         -1.001514         -1.140362 
##  magnet_forearm_z magnet_dumbbell_y   gyros_forearm_x     magnet_belt_y 
##         -1.221061         -1.809787         -1.923786         -2.235841 
##  gyros_dumbbell_x 
##       -126.321221
```

Unfortuantely no suitable data transoformation was identifued to reduce the skeness of the data. Neitehr `log` nor `BoxCox` trnasformations are applicable for where values are negaive. `expoTrans` transformation resulted in some `Inf` values across several predictors.

#####Data filtering
We have analyzed for near zero variance across the features:

```r
nearZeroVar(trn)
```

```
## integer(0)
```
No near zero varaimce has been identifed within remaning data.



```r
high.cor <- findCorrelation(cor(trn[, -ncol(trn)]), 0.90)
trn <- trn[, -high.cor]
tst <- tst[, -high.cor]
```

#####Data slicing

###Summary

###References
