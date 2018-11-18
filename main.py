import textFeatures
import csv
import numpy as np
import pandas as pd

# open files of text by Oscar Wilde (OW)

textA1 = open('OW-1.txt').read()
textA2 = open('OW-2.txt').read()
textA3 = open('OW-3.txt').read()
textA4 = open('OW-4.txt').read()

# open files from text by RWE
textB1 = open('RWE-1.txt').read()
textB2 = open('RWE-2.txt').read()
textB3 = open('RWE-3.txt').read()
textB4 = open('RWE-4.txt').read()

# put text into array
authorA = [textA1, textA2, textA3, textA4]
authorB = [textB1, textB2, textB3, textB4]

# create a csv file to write data values to
datafile = open('dataset.csv', 'w+')
writer = csv.writer(datafile)
writer.writerow(["sameAuthor", "lengthdiff", "stddevdiff", "richnessdiff"])


# comparing same author texts, adding to csv file with '1' for first column
for i in range(0, 3):
    for j in range(i+1, 4):
        writer.writerow(textFeatures.compare_features_same(authorA[i],authorA[j]))
        writer.writerow(textFeatures.compare_features_same(authorB[i],authorB[j]))


# compare diff author texts, adding to csv file with '0' for first column
for i in range(0,4):
    for j in range(0,4):
        writer.writerow(textFeatures.compare_features_diff(authorA[i], authorB[j]))

datafile.close()

#naivebayes implementation

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


dataframe = pd.read_csv("dataset.csv", sep=',')

X_train, X_test = train_test_split(dataframe, test_size=0.5)

gnb = GaussianNB()

features = ['lengthdiff','stddevdiff', 'richnessdiff']

gnb.fit(
    X_train[features].values,
    X_train['sameAuthor']
)
y_pred = gnb.predict(X_test[features])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test['sameAuthor'] != y_pred).sum(),
          100*(1-(X_test['sameAuthor'] != y_pred).sum()/X_test.shape[0])
))
