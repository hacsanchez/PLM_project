
# Human Activity Recognition
====================================================
## Practical Machine Learning
----------------------------------------------------
*Harold Cruz-Sanchez*

**Executive Summary:**

Base on the hypothesis that data generated by wearable-sensors could be use to find patters of human activity, a practical machine learning alghoritm (PML) was developed, trained, verificated and tested with data generated by four sensors, in order to find patters of human activity (Weight Lifting Exercises) of a subject wearing those sensors. 

The data collected contains 5 different classes of activities, (sitting-down, standing-up, standing, walking, and sitting), and was collected on 8 hours of activities of 4 healthy subjects. The raw data was processed, synchronized, code-labeled `classe`, while the participants were supervised by an experienced weight lifter.

Read more: http://groupware.les.inf.puc-rio.br/har

After downloading the data sets, a Random Forrest model was developed to reach a **high Accuracy (0.9937)** and **low error rate (0.6%)**.

The final model was able to correctly classify all 20 cases in the test set.


##Data 

This dataset is licensed under the **Creative Commons license (CC BY-SA)**. The CC BY-SA license means you can remix, tweak, and build upon this work even for commercial purposes, as long as you credit the authors of the original work and you license your new creations under the identical terms we are licensing to you.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3mcWM4mn1

The data for this project come from this source: 
http://groupware.les.inf.puc-rio.br/har

The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Thanks a lot to the authors

*Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.*

