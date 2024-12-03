import numpy as np
import flask
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from mne import EpochsArray, create_info # Import from mne after installation

stress_data = pd.read_csv('stress.csv')
emotion_data = pd.read_csv('heart_rate_emotion_dataset.csv')

print("Stress Dataset Head:\n", stress_data.head())
print("Emotion Dataset Head:\n", emotion_data.head())

X_stress = stress_data.drop(columns=['timestamps', 'Emotion', 'Subject'])
y_stress = stress_data['Emotion']

X_emotion = emotion_data.drop(columns=['Emotion'])
y_emotion = emotion_data['Emotion']

X_stress_train, X_stress_test, y_stress_train, y_stress_test = train_test_split(X_stress, y_stress, test_size=0.2, random_state=42)
X_emotion_train, X_emotion_test, y_emotion_train, y_emotion_test = train_test_split(X_emotion, y_emotion, test_size=0.2, random_state=42)

scaler_stress = StandardScaler()
X_stress_train = scaler_stress.fit_transform(X_stress_train)
X_stress_test = scaler_stress.transform(X_stress_test)

scaler_emotion = StandardScaler()
X_emotion_train = scaler_emotion.fit_transform(X_emotion_train)
X_emotion_test = scaler_emotion.transform(X_emotion_test)

stress_model = RandomForestClassifier(n_estimators=100, random_state=42)
stress_model.fit(X_stress_train, y_stress_train)

y_stress_pred = stress_model.predict(X_stress_test)
stress_accuracy = accuracy_score(y_stress_test, y_stress_pred)
stress_report = classification_report(y_stress_test, y_stress_pred)

print("Stress Dataset Results:")
print("Accuracy:", stress_accuracy)
print("Classification Report:\n", stress_report)

emotion_model = RandomForestClassifier(n_estimators=100, random_state=42)
emotion_model.fit(X_emotion_train, y_emotion_train)

y_emotion_pred = emotion_model.predict(X_emotion_test)
emotion_accuracy = accuracy_score(y_emotion_test, y_emotion_pred)
emotion_report = classification_report(y_emotion_test, y_emotion_pred)

print("\nEmotion Dataset Results:")
print("Accuracy:", emotion_accuracy)
print("Classification Report:\n", emotion_report)

info = create_info(ch_names=list(X_stress.columns), sfreq=128, ch_types='eeg')
X_train_reshaped = X_stress_train.reshape(X_stress_train.shape[0], X_stress_train.shape[1], 1)
X_test_reshaped = X_stress_test.reshape(X_stress_test.shape[0], X_stress_test.shape[1], 1)
epochs_train = EpochsArray(X_train_reshaped, info)
epochs_test = EpochsArray(X_test_reshaped, info)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_stress_train, y_stress_train)

model.fit(X_stress_train, y_stress_train)
y_pred = model.predict(X_stress_test)
accuracy = accuracy_score(y_stress_test, y_pred)


print(f"Accuracy: {accuracy}")
print("Classification Report:\n", emotion_report)

