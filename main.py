import simplifyBook
import textFeatures
import csv
import panda as pd

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

#naivebayes implementation
data = pd.read_csv('dataset.csv')
