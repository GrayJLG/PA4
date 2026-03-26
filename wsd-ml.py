# This program uses scikit learn to train a machine learning model of choice to perform word sense disambiguation using bag of words feature representation. If that last sentence could have
# been in klingon, we can define each:

# Word sense disambiguation is the process of using the context surrounding a word to try and determine its true meaning. In this program we try
# to determine if the word "line(s)" is referring to a phone line or a product line using the context of the surrounding sentence(s).

# bag of words feature representation is the process of associating all of the other words surrounding the disambigous word with how the word was meant to be used. We can 
# purposefully remove words that don't add much context using a "stoplist" which you can reference in the "bow" function below. 

# The machine learning models used to learn these associations and apply them are detailed in the input instructions below

# For input, the statement for running the program differs depending on the model you want to use. The program can run one of four models at at a time, starting with:

# MULTIMODAL NAIVE BAYES MODEL INPUT INSTRUCTIONS:
# python3 wsd-ml.py line-train.txt line-test.txt > my-line-answers.txt
# python3 wsd-ml.py line-train.txt line-test.txt MNaiveBayes > my-line-answers.txt
# Can both be used to train and test a multimodal Naive Bayes model. This is the default model, so it doesn't have to be specified unless you would like to.

# Multimodal Naive Bayes works by looking at the words surrounding the ambiguous word when training. During testing, it will check the surrounding context and
# predict the sense that was most associated with the surrounding context in each instance. 

# COMPLIMENT NAIVE BAYES MODEL INPUT INSTRUCTIONS:
# python3 wsd-ml.py line-train.txt line-test.txt CNaiveBayes > my-line-answers.txt

# Compliment Naive Bayes does the opposite of Multimodal Naive Bayes (go figure) by trying to determine what words are more likely in all other senses than the one it predicts.
# So if we predict product, we are saying that the context lacked words commonly associated with phone. This differs from multimodal in the sense that multimodal would predict product
# because many words in the context were associated with product. 

# LOGISTIC REGRESSION MODEL INPUT INSTRUCTIONS:
# python3 wsd-ml.py line-train.txt line-test.txt LogisticRegression > my-line-answers.txt

# Logistic regression works by assigning a weight to all contextual words in relation to their sense. It does not do this using probability like naive bayes does. 
# The weight of all words it was trained on that appear in the testing context are summed, and the result informs which sense is predicted. 

# SUPPORT VECTOR CLASSIFIER MODEL INPUT INSTRUCTIONS:
# python3 wsd-ml.py line-train.txt line-test.txt SVC > my-line-answers.txt

# SVC works by creating a divider in vector space that divides all of the bag of words sets it was trained on graphically. Then when testing it, the context is given a graphical point
# and whichever side of the line the point falls on determines the predicted sense.   

# Using any one of the models will generate the same output:

# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="phone"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="phone"/>
# <answer instance="line-n.w8_119:16927:" senseid="product"/>
# ...
# where each line indicates the context instance in the TEST data and your model's prediction for the sense of that instance

# as well as the accuracy and a confusion matrix for the model which is formatted as such:

# Output will be formatted as below:

# Accuracy: 93.65079365079364
#                 product phone
# product         49      5
# phone           3       69

# This is for the default (multimodal naive bayes), the numbers will differ for every model

# In terms of logic, the program works by first creating a list of bag of words and a list of associated labels. The bag of words lists are vectorized and a matrix is made where
# each row is a bag of words count vector and each column is a word from our vocabulary

# Then, using the scikit learn documentation we train the model of choice by passing it the word vector matrix and the list of labels from the training data

# To test our model, we first transform the test context into a feature vector using the same vocabulary as our training data. We then check that feature vector against the ones we 
# generated during training to find the sense that best matches. 

# The results are stored in my-line-answers.txt and are used in scorer.py to generate the accuracy and confusion matrix of the model. More information on how scorer.py works can be found in
# its header.

# Below is an analysis of the accuracy and confusion matrices for each of the integrated ml models

# ACCURACY AND CONFUSION MATRIX FOR MULTIMODAL NAIVE BAYES:
# Accuracy: 92.85714285714286, outperforms the MFS baseline of 42.857142857142854 by ~50%
# Confusion matrix:
#                 product phone
# product         49      5
# phone           4       68

# ACCURACY AND CONFUSION MATRIX FOR COMPLIMENT NAIVE BAYES:
# Accuracy: 93.65079365079364, outperforms the MFS baseline of 42.857142857142854 by ~50%
# Confusion matrix:
#                 product phone
# product         49      5
# phone           3       69

# ACCURACY AND CONFUSION MATRIX FOR LOGISTIC REGRESSION:
# Accuracy: 92.85714285714286, outperforms the MFS baseline of 42.857142857142854 by ~51%
# Confusion matrix:
#                 product phone
# product         51      3
# phone           6       66

# ACCURACY AND CONFUSION MATRIX FOR SUPPORT VECTOR CLASSIFIER:
# Accuracy: 93.65079365079364, outperforms the MFS baseline of 42.857142857142854 by ~51%
# Confusion matrix:
#                 product phone
# product         51      3
# phone           5       67

# DECISION LIST ACCURACY AND CONFUSION MATRIX FROM PA3 FOR COMPARISON:
# Accuracy: 81.74603174603175 which outperforms MFS baseline of 42.857142857142854 by ~39%
# Confusion matrix:
#                 product phone
# product         33      21
# phone           2       70

# We can see that all models integrated have an accuracy in the low 90s which are all improvements over the decision list model from PA3. Differences between machine learning models
# is always one or two instances being classified differently but the overall accuracy being consistently similar. Complement Naive Bayes and the Support Vector Classifier achieve the 
# highest accuracy at ~93.65%, while Multinomial Naive Bayes and Logistic Regression perform slightly lower at ~92.86%, indicating that all models are performing similarly with only minor 
# differences in classification decisions. The confusion matrices show that most errors occur when predicting the product sense, which is occasionally misclassified as phone, whereas 
# the phone sense is generally predicted more reliably.

# This program was written by Jacob Gray for CMSC 437 taught by Dr. Bridget McInness at Virginia Commonwealth University
# Last Update: 3/26/2026

# This website was instrumental for figuring out how all of the different models' functions worked. I linked their individual pages with their import statements below. 
# https://scikit-learn.org/stable/

import sys # for parsing user input
import re # for regex
import math # for log functions
from collections import defaultdict # for bag of words

# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes model 

# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html
from sklearn.naive_bayes import ComplementNB

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
from sklearn.linear_model import LogisticRegression # logistic regression model

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
from sklearn.svm import LinearSVC # linear support vector classifier model

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer # for vectorizing text data into numbers that the scikit models can use  


# Function for creating a list of contexts and senses from the training data to pass into the model of choice
def bow(trainText):

    contexts = []
    labels = []

    # Everything from here until the last 'fullContext = ...' statement is identical to my logic from PA3.
    with open (trainText, 'r', encoding ='utf-8') as file:
        instanceBlocks = re.findall(r"<instance.*?</instance>", file.read().lower(), re.DOTALL) # Grabs the raw text from each instance in the training data

    for instance in instanceBlocks:
        sense = re.search(r'senseid="(.*?)"', instance).group(1) # To make sense a string and not a list
        fullContext = re.search(r"<context>(.*?)</context>", instance, re.DOTALL).group(1) # To make the full context a string and not a list. 
        fullContext = re.sub(r"<head>", "", fullContext) # remove the starting head tag(s) around line/lines
        fullContext = re.sub(r"</head>", "", fullContext) # remove the trailing head tag(s) around line/lines

        # Make lists of the contexts and labels to give to the model of choice
        contexts.append(fullContext)
        labels.append(sense)

    return contexts, labels

# Function for training the multimodal naive bayes model
def mnbTrain(contexts, labels):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this builds our context vocabulary, removing stop words
    wordMat = vec.fit_transform(contexts) # transforms our vocabulary into matrix where each row is a bag of words count vector and each column is a word in our vocabulary

    # make the naive bayes model and fit it to our feature vectors and their labels
    mnb = MultinomialNB()
    mnb.fit(wordMat, labels)

    return mnb, vec

# Function for training the compliment naive bayes model
def cnbTrain(contexts, labels):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this builds our context vocabulary, removing stop words
    wordMat = vec.fit_transform(contexts) # transforms our vocabulary into matrix where each row is a bag of words count vector and each column is a word in our vocabulary

    # make the naive bayes model and fit it to our feature vectors and their labels
    cnb = ComplementNB()
    cnb.fit(wordMat, labels)

    return cnb, vec

# Function for training the logistic regression model
def lrTrain(contexts, labels):
    
    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this builds our context vocabulary, removing stop words
    wordMat = vec.fit_transform(contexts) # transforms our vocabulary into matrix where each row is a bag of words count vector and each column is a word in our vocabulary

    # make the logistic regression model and fit it to our feature vectors and their labels
    lr = LogisticRegression(max_iter=1000) # set this high so that there is a limit but we probably wont get there
    lr.fit(wordMat, labels)

    return lr, vec

# Function for training the support vector classifier model
def svcTrain(contexts, labels):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this builds our context vocabulary, removing stop words
    wordMat = vec.fit_transform(contexts) # transforms our vocabulary into matrix where each row is a bag of words count vector and each column is a word in our vocabulary

    # make the support vector classifier model and fit it to our feature vectors and their labels
    svc = LinearSVC()
    svc.fit(wordMat, labels)

    return svc, vec

def modelApply(model, vec, inFile):

    # Everything from here until the last 'fullContext = ...' statement is identical to my logic from PA3.
    with open(inFile, 'r', encoding='utf-8') as file:
        instanceBlocks = re.findall(r"<instance.*?</instance>", file.read().lower(), re.DOTALL) # grab all the instances in the same way we did last time
    
    for instance in instanceBlocks:
        prediction = ""
        instanceID = re.search(r'id="(.*?)"', instance).group(1) # find the instance id for output formatting
        fullContext = re.search(r"<context>(.*?)</context>", instance, re.DOTALL).group(1) # make sense a string and not a list
        fullContext = re.sub(r"<head>", "", fullContext) # remove the head tag
        fullContext = re.sub(r"</head>", "", fullContext) # remove the head tag


        wordMat = vec.transform([fullContext]) # The test sentence is converted into a feature vector using the same vocabulary as the training data.
        prediction = model.predict(wordMat)[0] # uses that vector to predict which word sense best matches the learned patterns.

        # prediction = "product" (used for MSF)

        # Print results in the same format as the key file
        print(f'<answer instance="{instanceID}" senseid="{prediction}"/>')

def main():

    # grab input files
    trainFile = sys.argv[1] 
    testFile = sys.argv[2]

    # Grab contexts and labels for bag of words
    contexts, labels = bow(trainFile)

    # determine desired model, default to naive bayes if none specified
    if(len(sys.argv) > 3):
        if(sys.argv[3] == "MNaiveBayes"):
            model, vec = mnbTrain(contexts, labels)
        elif(sys.argv[3] == "CNaiveBayes"):
            model, vec = cnbTrain(contexts, labels)
        elif(sys.argv[3] == "LogisticRegression"):
            model, vec = lrTrain(contexts, labels)
        elif(sys.argv[3] == "SVC"):
            model, vec = svcTrain(contexts, labels)
        else:
            # use sys.stderr so we can print to console, otherwise this gets added to my-line-answers.txt
            print("Model name not recognized, performing Multimodal Naive Bayes by default", file=sys.stderr)
            model, vec = mnbTrain(contexts, labels)
    else:
        model, vec = mnbTrain(contexts, labels)
    
    # apply the model to the test data
    modelApply(model, vec, testFile)

if __name__ == "__main__":
    main()
