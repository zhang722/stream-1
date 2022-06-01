# SPARK 2022 Utils

This repository contains all you need to start playing around with the SPARK Dataset.

Please first create a `data/` folder at the root of this project, then download the training and validation datasets in the newly created folder (see dedicated email for download link). After unziping the `*.zip` archives, the tree structure of `data/` must finally follow the one below:

<pre>
└───data/  
    ├───train_rgb/  
        ├───AcrimSat/  
        ├───Aquarius/  
        ├───Aura/  
        ├───...
    ├───train_depth/  
        ├───AcrimSat/  
        ├───Aquarius/  
        ├───Aura/  
        ├───...
    ├───train_labels.csv
    ├───validate_rgb/  
        ├───AcrimSat/  
        ├───Aquarius/  
        ├───Aura/  
        ├───...
    ├───validate_depth/  
        ├───AcrimSat/  
        ├───Aquarius/  
        ├───Aura/  
        ├───... 
    ├───validate_labels.csv
</pre>

The `visualize_data.ipynb` notebook contains basic functions to load and display dataset samples.

## SPARK Challenge evaluation

### Important dates

* Testing dataset release: 07/06/2021.
* Results submission deadline: 10/06/2021, 23:59 CET.
* Reports submission deadline: 17/06/2021, 23:59 CET.

### Results submission

Once the testing dataset is released (07/06), each participant will be able to fill the `test_labels_participant.csv` file with its model predictions.

This exact file must be sent to the Challenge email address `SPARK2021@uni.lu` before the results submission deadline (10/06, 23:59 CET). Prior to being sent, the file must be renamed as `test_labels_firstname-lastname-subX.csv`, where `firstname` and `lastname` must be replaced by the participant's first and last names, and `X` must be the index (in 1-digit format) of the submission: `X`=`1`, `2`, `3`, `4` or `5`.

Participants interested only in Task 1 can fill only the `class` column. Participants interested in Task 2 must fill both `class` and `bbox` columns.

The `class` column must be filled with class indexes:

| Class                     | Index    |
|---------------------------|----------|
| Proba 2                   | 0        |
| Cheops                    | 1        |
| Debris                    | 2        |
| Double star               | 3        |
| Earth Observation Sat 1   | 4        |
| Lisa Pathfinder           | 5        |
| Proba 3 CSC               | 6        |
| Proba 3 OCS               | 7        |
| Smart 1                   | 8        |
| Soho                      | 9        |
| Xmm Newton                | 10       |

The `bbox` cells must follow the format: `[R_min,C_min,R_max,C_max]`, where `R` refers to *row*, and `C` refers to *column*.

<img src="src/SPARK_bbox.PNG" alt="Bounding box format" width="300"/>

Up to 5 submissions are allowed per participant. However, please note that, for each task, only the best performing submission will be eligible for the prizes.

Every submissions received after the deadline, or that does not meet the required format, will be ignored.

### Evaluation metrics

Each task will be evaluated based on a dedicated metric.

#### Task 1: Classification

In this task, potential errors will be considered according to three different levels of seriousness:
* Classify a satellite as a wrong satellite: level 1/4.
* Classify a satellite as a debris: level 2/4.
* Classify a debris as a satellite: level 4/4.

Therefore, the following metric will be used to rank the participants:

<img src="src/SPARK_classif_metric.PNG" alt="Classification metric" width="300"/>

where

<img src="src/SPARK_F2-score.PNG" alt="F2-score" width="200"/>

and where <img src="src/SPARK_classif_metric_2.PNG" width="120"/> refers to the proportion of correctly classified samples (here applied to samples that actually belong to the 10 non-debris classes). 

#### Task 2: Detection

For this task, localization accuracy (bounding box) will be evaluated in addition to classification performance. The metric is largely inspired by the COCO Challenge one.

More precisely, we are going to compute the proportion of *correctly* predicted images, where a *correct* prediction refers to an image for which the predicted class is correct and the intersection-over-union (IoU) score between predicted and groundtruth bounding boxes is above a certain threshold. Finally, we are going to average these proportions over different IoU thresholds, to give more importance to more accurate results.

### Reports submission

A short report (2 or 3 pages, double columns, IEEE ICIP format), describing the methods used to generate the submitted results, must be sent by each participant before the reports submission deadline (17/06, 23:59 CET). This is mandatory to be eligible to the prizes.

## SPARK Dataset samples

<img src="https://drive.google.com/uc?id=1mgSt1Z230LbfijzDi1FIckQiEgBOReHD">
