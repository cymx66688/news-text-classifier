import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score


nrows = 200000
train_num = int(nrows * 0.8)
max_features = 3000

train_df = pd.read_csv('../input/train_set.csv', sep='\t', nrows=nrows)

vectorizer = CountVectorizer(max_features=max_features)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:train_num], train_df['label'].values[:train_num])

val_pred = clf.predict(train_test[train_num:])
print('Bag-of-Words+RidgeClassifier f1_score: {}'.format(f1_score(train_df['label'].values[train_num:], val_pred, average='macro')))
