import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
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

############################# Random Forest ##################################
print("############################# Random Forest ##################################")

X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=1000, random_state=10)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy pierwsza próba: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

X = df[['Accelerometer_x', 'Accelerometer_y', 'Accelerometer_z', 'Gyroscope_x', 'Gyroscope_y', 'Gyroscope_z']]
y = df['Label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))