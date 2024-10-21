import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

file_path = '01_Data_Processed.csv'
df = pd.read_csv(file_path)
print(df.head())

df['epoch (ms)'] = pd.to_datetime(df['epoch (ms)'], errors='coerce')
print(df.head())
# ============================== Plotting all values ==============================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Accelerometer_x'], bins=50, ax=axes[0], color='r').set_title('Accelerometer X')
sns.histplot(df['Accelerometer_y'], bins=50, ax=axes[1], color='g').set_title('Accelerometer Y')
sns.histplot(df['Accelerometer_z'], bins=50, ax=axes[2], color='b').set_title('Accelerometer Z')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Gyroscope_x'], bins=50, ax=axes[0], color='r').set_title('Gyroscope X')
sns.histplot(df['Gyroscope_y'], bins=50, ax=axes[1], color='g').set_title('Gyroscope Y')
sns.histplot(df['Gyroscope_z'], bins=50, ax=axes[2], color='b').set_title('Gyroscope Z')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Participants'], bins=50, ax=axes[0], color='r').set_title('Participants')
sns.histplot(df['Label'], bins=50, ax=axes[1], color='g').set_title('Label')
sns.histplot(df['Category'], bins=50, ax=axes[2], color='b').set_title('Category')
plt.tight_layout()
plt.show()

plt.hist(df['Set'], bins=50, color='b')
plt.title("Set")
plt.show()

# ============================== CORELATION MATRIX ==============================

numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

print("############################# Random Forest ##################################")

# Feature and label selection
X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
Random_forest_model = RandomForestClassifier(n_estimators=10, random_state=10)
Random_forest_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = Random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy pierwsza próba: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Random Forest)')
plt.show()

# ROC Curve for Multiclass
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predictions and probability estimates
y_prob = Random_forest_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Random Forest')
plt.legend(loc="lower right")
plt.show()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(Random_forest_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Random Forest): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

# Train Random Forest model with more estimators
Random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
Random_forest_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = Random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix for n_estimators=100
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Random Forest n_estimators=100)')
plt.show()

# Cross-validation for n_estimators=100
scores = cross_val_score(Random_forest_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Random Forest n_estimators=100): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

print("############################# LOGISTIC REGRESSION ##################################")

# Train logistic regression model
Logistic_regression_model = LogisticRegression()
Logistic_regression_model.fit(X_train, y_train)

# Predictions
y_pred = Logistic_regression_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Logistic Regression)')
plt.show()

# ROC Curve for Multiclass
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predictions and probability estimates
y_prob = Logistic_regression_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Multiclass Logistic Regression')
plt.legend(loc="lower right")
plt.show()

# Cross-validation
scores = cross_val_score(Logistic_regression_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Logistic Regression): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

# Train OVR model
Logistic_regression_model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', random_state=100)
Logistic_regression_model_ovr.fit(X_train, y_train)

# Predictions
y_pred_ovr = Logistic_regression_model_ovr.predict(X_test)
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
print(f'Accuracy (OVR): {accuracy_ovr:.2f}')
print(classification_report(y_test, y_pred_ovr))

# Confusion Matrix for OVR
cm_ovr = confusion_matrix(y_test, y_pred_ovr)
disp_ovr = ConfusionMatrixDisplay(confusion_matrix=cm_ovr)
disp_ovr.plot()
plt.title('Confusion Matrix (Logistic Regression OVR)')
plt.show()

# Cross-validation for OVR
scores_ovr = cross_val_score(Logistic_regression_model_ovr, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Logistic Regression OVR): {scores_ovr}')
print(f'Srednia dokladnosc cross-validation OVR: {np.mean(scores_ovr):.2f}')

print("############################# KNN ##################################")

# Feature and label selection
X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = KNN_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (KNN)')
plt.show()

# ROC Curve for Multiclass
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predictions and probability estimates
y_prob = KNN_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - KNN')
plt.legend(loc="lower right")
plt.show()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(KNN_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (KNN): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

# Train KNN model with n_neighbors=1
KNN_model = KNeighborsClassifier(n_neighbors=1)
KNN_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = KNN_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (n_neighbors=1): {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix for n_neighbors=1
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (KNN n_neighbors=1)')
plt.show()

# Cross-validation for n_neighbors=1
scores = cross_val_score(KNN_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (KNN n_neighbors=1): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

print("############################# Decision Tree Classifier ##################################")

# Feature and label selection
X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model (max_depth=1)
Decision_tree = DecisionTreeClassifier(max_depth=1)
Decision_tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = Decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (max_depth=1): {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Decision Tree max_depth=1)')
plt.show()

# ROC Curve for Multiclass
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Predictions and probability estimates
y_prob = Decision_tree.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Decision Tree (max_depth=1)')
plt.legend(loc="lower right")
plt.show()

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(Decision_tree, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Decision Tree max_depth=1): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

# Train Decision Tree model
Decision_tree = DecisionTreeClassifier(max_depth=20, random_state=42, min_samples_split=5)
Decision_tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = Decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (max_depth=20): {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix (Decision Tree max_depth=20)')
plt.show()

# ROC Curve
y_prob = Decision_tree.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Decision Tree (max_depth=20)')
plt.legend(loc="lower right")
plt.show()

# Cross-validation
scores = cross_val_score(Decision_tree, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Decision Tree max_depth=20): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')
