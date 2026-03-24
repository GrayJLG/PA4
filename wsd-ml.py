# https://scikit-learn.org/stable/

import sys # for parsing user input
import re # for regex
import math # for log functions
from collections import defaultdict # for bag of words

# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
from sklearn.naive_bayes import MultinomialNB # Multinomial Naive Bayes model 

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
from sklearn.linear_model import LogisticRegression # logistic regression model

#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
from sklearn.svm import LinearSVC # linear support vector classifier model

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer # for vectorizing text data into numbers that the scikit models can use  


# Function for associating a bag of words with a sense for each context in the training data
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

        # Make lists of the contexts and labels to give the model of choice
        contexts.append(fullContext)
        labels.append(sense)

    return contexts, labels

# Function for training the naive bayes model
def nbTrain(contexts, labels):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this vectorizes our contexts for use in the models, and removes the same stopwords as before
    wordMat = vec.fit_transform(contexts) # matrix of all of the feature vectors

    # make the naive bayes model and fit it to our feature vectors and their labels
    nb = MultinomialNB()
    nb.fit(wordMat, labels)

    return nb, vec

def lrTrain(contexts, labels):
    
    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this vectorizes our contexts for use in the models, and removes the same stopwords as before
    wordMat = vec.fit_transform(contexts) # matrix of all of the feature vectors

    # make the logistic regression model and fit it to our feature vectors and their labels
    lr = LogisticRegression(max_iter=1000)
    lr.fit(wordMat, labels)

    return lr, vec

def svcTrain(contexts, labels):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = ["line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"]

    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vec = CountVectorizer(stop_words = stopwords) # this vectorizes our contexts for use in the models, and removes the same stopwords as before
    wordMat = vec.fit_transform(contexts) # matrix of all of the feature vectors

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


        wordMat = vec.transform([fullContext]) # make feature vector matrix based on the vector we trained the model on
        prediction = model.predict(wordMat)[0] # use the feature vector matrix to predict a sense for every feature vector

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
        if(sys.argv[3] == "NaiveBayes"):
            model, vec = nbTrain(contexts, labels)
        elif(sys.argv[3] == "LinearRegression"):
            model, vec = lrTrain(contexts, labels)
        elif(sys.argv[3] == "SVC"):
            model, vec = svcTrain(contexts, labels)
        else:
            # use sys.stderr so we can print to console, otherwise this gets added to my-line-answers.txt
            print("Model name not recognized, performing Naive Bayes by default", file=sys.stderr)
            model, vec = nbTrain(contexts, labels)
    else:
        model, vec = nbTrain(contexts, labels)
    
    # apply the model to the test data
    modelApply(model, vec, testFile)

if __name__ == "__main__":
    main()
