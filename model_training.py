import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("FakeNewsNet.csv")

X = df['title'] 
y = df['real']    

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer(stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print(" Model trained & saved!")
