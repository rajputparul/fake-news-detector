import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("FakeNewsNet.csv")

# # Data
X = df['title'] 
y = df['real']    

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)

# Model
model = LogisticRegression()
model.fit(x_train_vec, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model trained & saved!")