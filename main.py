import os
import re
import pickle
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv('amazon_alexa.tsv', delimiter='\t', quoting=3)

# Print dataset info
print(f"Dataset Shape: {data.shape}")
print(f"Feature Names: {data.columns.values}")

# Drop missing values
data.dropna(inplace=True)

# Add a new column 'length' (length of the review)
data['length'] = data['verified_reviews'].apply(len)

# Rating Distribution
data['rating'].value_counts().plot.bar(color='red')
plt.title('Rating Distribution Count')
plt.xlabel('Ratings')
plt.ylabel('Count')
plt.show()

# Pie chart for rating distribution
fig = plt.figure(figsize=(7, 7))
colors = ('red', 'green', 'blue', 'orange', 'yellow')
wp = {'linewidth': 1, "edgecolor": 'black'}
tags = data['rating'].value_counts() / data.shape[0]
explode = (0.1, 0.1, 0.1, 0.1, 0.1)
tags.plot(kind='pie', autopct="%1.1f%%", shadow=True, colors=colors, startangle=90, wedgeprops=wp, explode=explode)
plt.title('Percentage-wise distribution of ratings')
plt.show()

# Display a review with feedback 0 & 1 safely
if not data[data['feedback'] == 0].empty:
    review_0 = data[data['feedback'] == 0].iloc[0]['verified_reviews']
    print("Negative Review Example:", review_0)

if not data[data['feedback'] == 1].empty:
    review_1 = data[data['feedback'] == 1].iloc[0]['verified_reviews']
    print("Positive Review Example:", review_1)

# Feedback Distribution
data['feedback'].value_counts().plot.bar(color='blue')
plt.title('Feedback Distribution Count')
plt.xlabel('Feedback')
plt.ylabel('Count')
plt.show()

# Variation-wise Mean Ratings
data.groupby('variation')['rating'].mean().sort_values().plot.bar(color='brown', figsize=(11, 6))
plt.title("Mean Rating According to Variation")
plt.xlabel('Variation')
plt.ylabel('Mean Rating')
plt.show()

# Histogram of review lengths
sns.histplot(data['length'], color='blue').set(title='Distribution of Length of Reviews')

# Word Cloud
reviews = " ".join(data['verified_reviews'])
wc = WordCloud(background_color='white', max_words=50)
plt.figure(figsize=(10, 10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for All Reviews', fontsize=10)
plt.axis('off')
plt.show()

# Preprocessing reviews using stemming
corpus = []
stemmer = PorterStemmer()
for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['verified_reviews'])
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = ' '.join(review)
    corpus.append(review)

# Convert text data to numerical features
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y = data['feedback'].values

# Ensure model directory exists
if not os.path.exists('Models'):
    os.makedirs('Models')

# Save CountVectorizer
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Scale data
scaler = MinMaxScaler()
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Save Scaler Model
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))

# Train Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

# Evaluate Model
print("Random Forest Training Accuracy:", model_rf.score(X_train_scl, y_train))
print("Random Forest Testing Accuracy:", model_rf.score(X_test_scl, y_test))

# Predict on Test Set
y_preds = model_rf.predict(X_test_scl)

# Confusion Matrix
cm = confusion_matrix(y_test, y_preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_rf.classes_)
cm_display.plot()
plt.show()

# Cross Validation Accuracy
accuracies = cross_val_score(estimator=model_rf, X=X_train_scl, y=y_train, cv=10)
print("Cross Validation Accuracy:", accuracies.mean())
print("Standard Variance:", accuracies.std())

# Hyperparameter Tuning with Grid Search
params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}

cv_object = StratifiedKFold(n_splits=5)  # Increased from 2 to 5
grid_search = GridSearchCV(estimator=model_rf, param_grid=params, cv=cv_object, verbose=0, return_train_score=True)
grid_search.fit(X_train_scl, y_train.ravel())

# Train XGBoost Classifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)

# Predict using XGBoost
y_preds_xgb = model_xgb.predict(X_test_scl)  # FIXED: Used Scaled Data

# Confusion Matrix for XGBoost
cm_xgb = confusion_matrix(y_test, y_preds_xgb)
cm_display_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=model_xgb.classes_)
cm_display_xgb.plot()
plt.show()

# Save XGBoost Model
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))

# Train Decision Tree Classifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)

# Predict using Decision Tree
y_preds_dt = model_dt.predict(X_test_scl)  # FIXED: Used Scaled Data
