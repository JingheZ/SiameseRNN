# Representation Learning with Multi-Channel CNN with Attention


To do:
1. create vectors for each visit for each patient
2. Padding
3. Build CNN model (multi-channel vs. single-channel)


mGraph results:
# Baseline 1:
Threshold tuned with f measure, AUC: 0.561
             precision    recall  f1-score   support
          0       0.88      0.82      0.85      5340
          1       0.21      0.30      0.25       832
avg / total       0.79      0.75      0.77      6172
Threshold tuned with AUC, AUC: 0.568
             precision    recall  f1-score   support
          0       0.89      0.81      0.84      5340
          1       0.21      0.33      0.26       832
avg / total       0.79      0.74      0.77      6172

# Baseline 2:
Threshold tuned with f measure, AUC: 0.697
             precision    recall  f1-score   support
          0       0.92      0.94      0.93      5340
          1       0.53      0.46      0.49       832
avg / total       0.86      0.87      0.87      6172
Threshold tuned with AUC, AUC: 0.706
             precision    recall  f1-score   support
          0       0.92      0.91      0.91      5340
          1       0.46      0.51      0.48       832
avg / total       0.86      0.85      0.86      6172



