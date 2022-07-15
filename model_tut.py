import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")
# have tp specify that columns should be dropped with axis=1
# inplace=True changes the df itself and does not make a new copy
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.rename(columns={"v1": "class", "v2": "message"}, inplace=True)
# Create new label column
df['label'] = df['class'].map({'ham': 0, 'spam': 1})
x = df["message"]
y = df["label"]

#Create countVectorizer
cv = CountVectorizer()

#fit_transform requires an iterable which generates either str, unicode or file objects
# as input. 
# Apparently, <class 'pandas.core.series.Series'> from x = df["message"]
# satisfies this condition

#fit_transform does two things: 
# Found all of the different words in the text
# Counted how many of each there were
# array of shape (n_samples, n_features)
# Document-term matrix.


x = cv.fit_transform(x)

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# Split arrays or matrices into random train and test subsets
# sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = MultinomialNB()

# Trains model on training data
clf.fit(x_train,y_train)

# Returns float of correct predictions/all predictions for info
print(clf.score(x_test,y_test))

y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))

