# A module to analyze a machine learning model for bias.
This is largely based on the Claidry Monitors offered in the AWS SageMaker development kit. Here I am removing the bloat of containerizing the process and unpacking the functionality. The inspiration comes from the inability of the SageMaker Clarify monitors is the inability to analyze features or data that is not in the model. If we monitor bias on protected class data joined on the machine learning features, target, and ground truth value, we can uncover bias the may come from the correlation of different features; or, perhaps implict bias in real outcomes we may not otherwise know of.

This is a work in progress. At some point there will be a SageMaker pipeline step created out of this, wrapping this code in the a form that can generate a SageMaker pipeline step.

## There is some prerequsite work users need
- Users need to determine what features are bias
- Users need to logically seperate the groups into 2 classes, an advantaged group and a disadvantaged group
- This is support for the basic model types
    - linear regression
    - logistic regression/binary classification
    - multi-class classification

## Both pre-training and post training bias is supported
- Pre training bias analyzes the raw data for inherent bias
- Post training bias analyzes the model predictions for bias


## Seperating facets
There is some prework required to seperate data examples into distinct demographic groups. This partition of groups represents a favored and disfavored group. If the feature value of interest is categorical in nature, ie a lable, this goup can simply be partitioned into an advantaged or disadvantaged group by assign one of the class labels. For a continuous value, a threshold needs to be determined that logically seperates data examples into a facored and disfavored group. One can try different thresholds and see where the highest divergence is between loss function values to determine this. The central idea to consider in the set up is partitioning demographics into different groups and for the post training analysis, model scores, the outcomes into 2 distinct group. 
## TODO: add metrics and breakdown
