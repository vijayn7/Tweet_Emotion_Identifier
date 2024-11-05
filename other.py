import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_parquet('training_data.parquet')
emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
data['emotion'] = data['label'].map(emotion_map)

# Plot the distribution of emotions
plt.figure(figsize=(10, 6))
sns.countplot(data['emotion'])
plt.title('Distribution of Emotions')
plt.ylabel('Emotion')
plt.xlabel('Count')
plt.show()

# Splitting the data
X = data['text']  # Feature: tweet text
y = data['label']  # Target: emotion labels (0-5)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display the classification report
print(classification_report(y_test, y_pred, target_names=emotion_map.values()))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_map.values(), yticklabels=emotion_map.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot the most important features for each emotion
def plot_top_features(vectorizer, model, n=10):
    feature_names = vectorizer.get_feature_names_out()
    for i, emotion in emotion_map.items():
        top_features = model.coef_[i].argsort()[-n:]
        plt.figure(figsize=(10, 6))
        plt.barh(range(n), model.coef_[i][top_features], align='center')
        plt.yticks(range(n), [feature_names[j] for j in top_features])
        plt.xlabel('Importance')
        plt.title(f'Top {n} Features for {emotion}')
        plt.show()

plot_top_features(vectorizer, model)