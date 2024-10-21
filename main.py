import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

file_path = '01_Data_Processed.csv'
df = pd.read_csv(file_path)
print(df.head())

df['epoch (ms)'] = pd.to_datetime(df['epoch (ms)'], errors='coerce')
print(df.head())
################################## PLOTING ALL VALUES ##################################

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

#############################CORELATION MATRIX##################################

numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

print("############################# Random Forest ##################################")

X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
Random_forest_model = RandomForestClassifier(n_estimators=10, random_state=10)
Random_forest_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = Random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy pierwsza próba: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(Random_forest_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Random Forest): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Train the model
Random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
Random_forest_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = Random_forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(Random_forest_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Random Forest n_estimators=100): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

print("############################# LOGISTIC REGRESION ##################################")

Logistic_regreation_model = LogisticRegression()
Logistic_regreation_model.fit(X_train, y_train)

y_pred = Logistic_regreation_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(Logistic_regreation_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Logistic Regression): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

Logistic_regreation_model = LogisticRegression(multi_class='ovr', solver='lbfgs', random_state=100)
Logistic_regreation_model.fit(X_train, y_train)

y_pred = Logistic_regreation_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(Logistic_regreation_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Logistic Regression ovr): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

print("############################# KNN ##################################")

KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
y_pred = KNN_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(KNN_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (KNN): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

KNN_model = KNeighborsClassifier(n_neighbors=1)
KNN_model.fit(X_train, y_train)
y_pred = KNN_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(KNN_model, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (KNN n_neighbors=1): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

print("############################# Decision Tree Classifier ##################################")

Decision_tree = DecisionTreeClassifier(max_depth=1)
Decision_tree.fit(X_train, y_train)
y_pred = Decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(Decision_tree, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Decision Tree max_depth=1): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')

Decision_tree = DecisionTreeClassifier(max_depth=20, random_state=42, min_samples_split=5)
Decision_tree.fit(X_train, y_train)
y_pred = Decision_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Cross-validation
scores = cross_val_score(Decision_tree, X, y, cv=cv, scoring='accuracy')
print(f'Cross-validation scores (Decision Tree max_depth=20): {scores}')
print(f'Srednia dokladnosc cross-validation: {np.mean(scores):.2f}')
