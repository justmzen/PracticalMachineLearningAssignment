Introduction
============

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, the goal is
to use data from accelerometers on the belt, forearm, arm, and dumbell
of 6 participants. They were asked to perform barbell lifts correctly
and incorrectly in 5 different ways.

The goal of this project is to predict the manner in which a participant
did the exercise. This is the “classe” variable in the training set. Any
of the other variables can be used to predict with.

A report should be created describing how the model was built, how cross
validation was used, the expected out of sample error and the reasons
for all those choices. It is also asked to used the prediction model to
predict 20 different test cases.

The authors provide a data set with 5 classes (sitting-down,
standing-up, standing, walking, and sitting) of 4 subjects in a range of
8 hours activities \[1\].

The data for the project come from this source:
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>.

More information is available from the website here:
<a href="http://groupware.les.inf.puc-rio.br/har" class="uri">http://groupware.les.inf.puc-rio.br/har</a>
(see the section on the Weight Lifting Exercise Dataset).

Data downloading a reading
==========================

    if(!dir.exists('./input')) dir.create('./input')
    if(!file.exists(('./input/training.csv'))){
            fileURLTraining <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
            download.file(fileURLTraining, destfile = './input/training.csv')
            rm(fileURLTraining)
    }
    if(!file.exists(('./input/testing.csv'))){
            fileURLTesting <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
            download.file(fileURLTesting, destfile = './input/testing.csv')
            rm(fileURLTesting)
    }

    global_training <- read.csv(file = './input/training.csv',
                                stringsAsFactors = TRUE)
    validation <- read.csv(file = './input/testing.csv',
                           stringsAsFactors = TRUE)

Exploratory data analysis
=========================

In order to find relevant variables we will perform a brief EDA. There
are 160 variables and 19622 observations.

The first 5 variables are label variables, such as row number and
people’s names, whereas all the other from the 6th to the 160th are
logical or numeric:

    head(global_training)[1:10]

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
    ## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
    ## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
    ## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
    ## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
    ##   new_window num_window roll_belt pitch_belt yaw_belt
    ## 1         no         11      1.41       8.07    -94.4
    ## 2         no         11      1.41       8.07    -94.4
    ## 3         no         11      1.42       8.07    -94.4
    ## 4         no         12      1.48       8.05    -94.4
    ## 5         no         12      1.48       8.07    -94.4
    ## 6         no         12      1.45       8.06    -94.4

We will exclude the first five variables in order to have only logical,
numerical and relevant factor variables.

    global_training <- global_training[, -(1:5)]
    validation <- validation[, -(1:5)]

There is also an important number of variables mostly composed by NAs.
We will exclude these variables as well.

    emptyVariables <- sapply(X = global_training,
                             function(x) {mean(is.na(x)) > 0.8}
    )

    global_training <- global_training[, emptyVariables == FALSE]
    validation <- validation[, emptyVariables == FALSE]

    numNAVariables <- sum(emptyVariables)

The number of these variables is 67.

There are also variables with low variability, hence not representative
of the phenomenons. We will also exclude them. The ‘caret’ package
provides the ‘nearZeroVar’ function which diagnoses predictors with zero
variance and predictors with few unique values compared to the number of
samples.

    library('caret')
    nearZeroVarVariables <- nearZeroVar(global_training)

    global_training <- global_training[, -nearZeroVarVariables]
    validation <- validation[, -nearZeroVarVariables]

Cross validation
================

The large amount of data provided as training data let us split it into
two different data sets, one used as the trainer and called “training”
and the second used as the tester and called “testing”. The secondo
original data set, provided for the prediction, will be called
“validation”.

The partition of the original data set will be done via random sampling.
A percentage fixed at 70% will form the training set and the other 30%
will form the test one:

    set.seed(6574)
    inTrain <- createDataPartition(y = global_training$classe,
                                   p = 0.70, list = FALSE)

    training <- global_training[inTrain, ]
    testing <- global_training[-inTrain, ]

Machine learning modelling
==========================

Three machine learning models will be built on the ‘training’ data set
and then applied to the ‘testing’ data set. The accuracy will be
calculated for each of them through confusion matrices. The machine
leaning model with the highest accuracy will finally be applied to
predict the ‘class’ of the ‘validation’ data set.

1. Decision tree
----------------

    # Model building
    set.seed(98621)
    require('e1071')
    modFitDecisionTree <- train(classe ~ ., data = training,
                                method = 'rpart')

    # Plot
    require('rattle')
    fancyRpartPlot(modFitDecisionTree$finalModel, caption = '')

![](practical_machine_learning_assignment_files/figure-markdown_strict/Decision%20tree%20model-1.png)

    # Prediction
    predictionDecisionTree <- predict(modFitDecisionTree,
                                      newdata = testing)

    # Accuracy
    confDecisionTree <- confusionMatrix(predictionDecisionTree,
                                        testing$classe)
    confDecisionTree

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1528  473  483  428  108
    ##          B   21  377   30  196   87
    ##          C  118  289  513  299  215
    ##          D    0    0    0    0    0
    ##          E    7    0    0   41  672
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5251          
    ##                  95% CI : (0.5122, 0.5379)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.3797          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9128  0.33099  0.50000   0.0000   0.6211
    ## Specificity            0.6457  0.92962  0.81045   1.0000   0.9900
    ## Pos Pred Value         0.5060  0.53024  0.35774      NaN   0.9333
    ## Neg Pred Value         0.9490  0.85273  0.88475   0.8362   0.9206
    ## Prevalence             0.2845  0.19354  0.17434   0.1638   0.1839
    ## Detection Rate         0.2596  0.06406  0.08717   0.0000   0.1142
    ## Detection Prevalence   0.5132  0.12082  0.24367   0.0000   0.1223
    ## Balanced Accuracy      0.7792  0.63031  0.65523   0.5000   0.8055

2. Random forest model
----------------------

    # Model building
    set.seed(98621)
    require('dplyr')
    require('randomForest')
    trControlRF <- trainControl(method = 'cv', number = 3, verboseIter = FALSE)
    startTime <- Sys.time()
    modFitRandomForest <- train(x = select(training, -classe),
                                y = as.factor(training$classe),
                                method = 'rf',
                                trControl = trControlRF)
    endTime <- Sys.time()

    # Prediction
    predictionRandomForest <- predict(modFitRandomForest,
                                      newdata = testing)

    # Accuracy
    confRandomForest <- confusionMatrix(predictionRandomForest,
                                        testing$classe)
    confRandomForest

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1674    8    0    0    0
    ##          B    0 1127    5    0    0
    ##          C    0    4 1021    2    0
    ##          D    0    0    0  962    3
    ##          E    0    0    0    0 1079
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9963          
    ##                  95% CI : (0.9943, 0.9977)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9953          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   0.9895   0.9951   0.9979   0.9972
    ## Specificity            0.9981   0.9989   0.9988   0.9994   1.0000
    ## Pos Pred Value         0.9952   0.9956   0.9942   0.9969   1.0000
    ## Neg Pred Value         1.0000   0.9975   0.9990   0.9996   0.9994
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2845   0.1915   0.1735   0.1635   0.1833
    ## Detection Prevalence   0.2858   0.1924   0.1745   0.1640   0.1833
    ## Balanced Accuracy      0.9991   0.9942   0.9969   0.9987   0.9986

    # Execution time
    exTime <- endTime - startTime

Generally, random forest models overfit and require long calculation
times. The time require for this model and for these parameters is 9.18
minutes with the following environment:

    sessionInfo()

    ## R version 4.0.3 (2020-10-10)
    ## Platform: x86_64-apple-darwin17.0 (64-bit)
    ## Running under: macOS Big Sur 10.16
    ## 
    ## Matrix products: default
    ## BLAS:   /Library/Frameworks/R.framework/Versions/4.0/Resources/lib/libRblas.dylib
    ## LAPACK: /Library/Frameworks/R.framework/Versions/4.0/Resources/lib/libRlapack.dylib
    ## 
    ## locale:
    ## [1] it_IT.UTF-8/it_IT.UTF-8/it_IT.UTF-8/C/it_IT.UTF-8/it_IT.UTF-8
    ## 
    ## attached base packages:
    ## [1] stats     graphics  grDevices utils     datasets  methods   base     
    ## 
    ## other attached packages:
    ## [1] randomForest_4.6-14 dplyr_1.0.2         rattle_5.4.0       
    ## [4] bitops_1.0-6        tibble_3.0.4        e1071_1.7-4        
    ## [7] caret_6.0-86        ggplot2_3.3.2       lattice_0.20-41    
    ## 
    ## loaded via a namespace (and not attached):
    ##  [1] tidyselect_1.1.0     xfun_0.19            purrr_0.3.4         
    ##  [4] reshape2_1.4.4       splines_4.0.3        colorspace_2.0-0    
    ##  [7] vctrs_0.3.5          generics_0.1.0       htmltools_0.5.0     
    ## [10] stats4_4.0.3         yaml_2.2.1           survival_3.2-7      
    ## [13] prodlim_2019.11.13   rlang_0.4.9          ModelMetrics_1.2.2.2
    ## [16] pillar_1.4.7         glue_1.4.2           withr_2.3.0         
    ## [19] RColorBrewer_1.1-2   foreach_1.5.1        lifecycle_0.2.0     
    ## [22] plyr_1.8.6           rpart.plot_3.0.9     lava_1.6.8.1        
    ## [25] stringr_1.4.0        timeDate_3043.102    munsell_0.5.0       
    ## [28] gtable_0.3.0         recipes_0.1.15       codetools_0.2-16    
    ## [31] evaluate_0.14        knitr_1.30           class_7.3-17        
    ## [34] Rcpp_1.0.5           scales_1.1.1         ipred_0.9-9         
    ## [37] digest_0.6.27        stringi_1.5.3        grid_4.0.3          
    ## [40] tools_4.0.3          magrittr_2.0.1       crayon_1.3.4        
    ## [43] pkgconfig_2.0.3      ellipsis_0.3.1       MASS_7.3-53         
    ## [46] Matrix_1.2-18        data.table_1.13.2    pROC_1.16.2         
    ## [49] lubridate_1.7.9.2    gower_0.2.2          rmarkdown_2.5       
    ## [52] iterators_1.0.13     R6_2.5.0             rpart_4.1-15        
    ## [55] nnet_7.3-14          nlme_3.1-149         compiler_4.0.3

3. Generalised boosted model
----------------------------

    # Model building
    set.seed(98621)
    require('gbm')
    trControlGBM <- trainControl(method = 'cv', number = 4, repeats = 1)
    modFitBoosting <- train(classe ~ ., data = training,
                            method = 'gbm', trControl = trControlGBM,
                            verbose = FALSE)

    # Prediction
    predictionBoosting <- predict(modFitBoosting,
                                  newdata = testing)

    # Accuracy
    confBoosting <- confusionMatrix(predictionBoosting,
                                    testing$classe)
    confBoosting

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1667   14    0    0    0
    ##          B    7 1114    9    3    6
    ##          C    0   10 1014   11    2
    ##          D    0    1    1  949   12
    ##          E    0    0    2    1 1062
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9866          
    ##                  95% CI : (0.9833, 0.9894)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.983           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9958   0.9781   0.9883   0.9844   0.9815
    ## Specificity            0.9967   0.9947   0.9953   0.9972   0.9994
    ## Pos Pred Value         0.9917   0.9781   0.9778   0.9855   0.9972
    ## Neg Pred Value         0.9983   0.9947   0.9975   0.9970   0.9959
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2833   0.1893   0.1723   0.1613   0.1805
    ## Detection Prevalence   0.2856   0.1935   0.1762   0.1636   0.1810
    ## Balanced Accuracy      0.9962   0.9864   0.9918   0.9908   0.9904

Model selection
---------------

Based on the overall accuracies:  
- Decision tree model: 0.5250637  
- Random forest model: 0.9962617  
- Boosting model: 0.986576

The random forest model will be used for the prediction.

Prediction
==========

With random forest model, the predicted values for the ‘validation’ data
set are:

    prediction <- predict(modFitRandomForest,
                    newdata = validation)
    prediction

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Bibliografy
===========

\[1\] Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.;
Fuks, H. Wearable Computing: Accelerometers’ Data Classification of Body
Postures and Movements. Proceedings of 21st Brazilian Symposium on
Artificial Intelligence. Advances in Artificial Intelligence - SBIA
2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR:
Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI:
10.1007/978-3-642-34459-6\_6.
