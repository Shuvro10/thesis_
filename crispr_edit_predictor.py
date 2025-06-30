import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('CIRCLE_seq_data.csv')

# Features and target
X = df['off_seq']
y = df['label']

# Convert sequences to numerical features (simple k-mer count vectorization)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))  # 3-mer example
X_vect = vectorizer.fit_transform(X)

# Split the data: 30% train, 70% test
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, train_size=0.3, random_state=42, stratify=y
)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))