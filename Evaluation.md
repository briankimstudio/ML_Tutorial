## Evaluation

### Confusion matrix

Confusion matrix shows how many predictions are correct or wrong. Typically, x axis is predicted label and y axis is true label(0 for negative, 1 for positive). 

For example, among values predicted as positive, 104 are correct(TP), but two are wrong(FP). Likewise, among values predicted as negative, 61 are correct(TN), but two are wrong(FN)

![confision matrix](/images/Confusion_matrix.png)

|   | Predicted Negative |Predicted Positive|
|---|:---:|:---:|
| Actual Negative | TN(61) | FP(2) |
| Actual Positive | FN(4) | TP(104) |

- Accuracy = (TP+TN) / (TP+TN+FP+FN)
- Precision = TP / (TP+FP)
- Recall = TP / (TP+FN) 

In `sklearn`, use these functions to calculate accuracy, precision, and recall.

```
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
```

### Receiver Operating Charasteristic(ROC) curve

In this curve, x axis indicates False Positive Rate(FPR) and y axis indicates True Positive Rate(TPR)

![confision matrix](/images/AUC.png)

- TPR(recall) = TP / (TP+FN)
- FPR = FP / (FP+TN)

### Area Under the ROC Curve(AUC)

AUC is a frequently used indicator showing the performance of the model. It ranges from 0 to 1 and higher is better. 
