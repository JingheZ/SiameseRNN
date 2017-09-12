# Representation Learning with Multi-Channel CNN with Attention


To do:
1. create vectors for each visit for each patient
2. Padding
3. Build CNN model (multi-channel vs. single-channel)


Hierarchical Representation of Longitudinal EHR
Baseline results:

Baseline 1: frequency 
Threshold 0.240 tuned with f5 measure, AUC: 0.599
             precision    recall  f1-score   support
          0       0.89      0.64      0.75      5534
          1       0.21      0.56      0.30       944
avg / total       0.79      0.63      0.68      6478
Threshold 0.420 tuned with AUC, AUC: 0.575
             precision    recall  f1-score   support
          0       0.88      0.78      0.82      5534
          1       0.22      0.37      0.28       944
avg / total       0.78      0.72      0.74      6478

Baseline 2: frequency in sub-window
Threshold 0.240 tuned with f measure, AUC: 0.760
             precision    recall  f1-score   support
          0       0.94      0.82      0.88      5534
          1       0.40      0.70      0.51       944
avg / total       0.86      0.80      0.82      6478
Threshold 0.410 tuned with AUC, AUC: 0.769
             precision    recall  f1-score   support
          0       0.93      0.93      0.93      5534
          1       0.60      0.61      0.60       944
avg / total       0.88      0.88      0.88      6478


To do:
1. add baselines
2. get the base classifier and filter weak classifiers
3. collect results of both parts

