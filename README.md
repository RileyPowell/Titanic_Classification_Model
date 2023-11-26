# Titanic_Classification_Model
This is a classification model built to determine passengers on the titanic that survive and die. This is one of the most used datasets, especially for begineers, when getting into machine learning.
The data requiered a small amount of data cleaning such as dropping columns, imputing median values for missing values, and encoding string values to integers. After cleaning the data is feed into multiple machine learning models to compare their performance. The models are evaluated using their accuracy and R2 score.

Accuracy: $\frac{TP + TN}{TP + TN + FP + FN}$
<sub>_Where TP is True Positive, TN is True Negative, FP is False Positive, and FN is False Negative._<sub\>

R2: also known as the coefficient of determination measures the variability in the dependent variable Y that is being explained by the independent variables Xi in the regression model.

The modesl evaluated are:
* Logistic Regression.
* Logistic Regression Neural Network.
* Naive Bayes.
* K Neighbors Classifier.
* Decision Tree.
* Support Vector Machine.
* A voting ensemble of all these models.
