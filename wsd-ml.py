# https://scikit-learn.org/stable/

import sys # for parsing user input
import re # for regex
import math # for log functions
from collections import defaultdict # for bag of words
from sklearn import MultinomialNB # Multinomial Naive Bayes model https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# Function for associating a bag of words with a sense for each context in the training data
def bow(trainText):

    # I found this list https://gist.github.com/sebleier/554280 and decided to pull some common ones from it that would likely appear in the training data. I found that messing
    # with this did not effect my accuracy or confusion matrix that much so I settled for this. I added the line/lines into here to avoid more regex shenanigans later.
    stopwords = {"line", "lines", "the", "is", "and", "if", "or", "but", "a", "an", "of", "to", "in", "on", "for", "with", "into"}

    # For storing sense to bag of words associations
    senseMap = defaultdict(list)

    with open (trainText, 'r', encoding ='utf-8') as file:
        instanceBlocks = re.findall(r"<instance.*?</instance>", file.read().lower(), re.DOTALL) # Grabs the raw text from each instance in the training data

    for instance in instanceBlocks:
        sense = re.search(r'senseid="(.*?)"', instance).group(1) # To make sense a string and not a list
        fullContext = re.search(r"<context>(.*?)</context>", instance, re.DOTALL).group(1) # To make the full context a string and not a list. 
        fullContext = re.sub(r"<head>", "", fullContext) # remove the starting head tag(s) around line/lines
        fullContext = re.sub(r"</head>", "", fullContext) # remove the trailing head tag(s) around line/lines

        tokens = set(re.findall(r"\w+", fullContext)) # tokenize the context and place it in a set to remove duplicate entries
        tokens = tokens - stopwords # remove the stopwords from the set
            
        senseMap[sense].append(tokens) # add sense + bag of words associations to the hashmap

    return senseMap

# Function for creating a decision list from the bag of words associations gathered from the training data
def dl(senseMap, modelFile):

    decList = [] # List of decision data
    wordPerSense = {} # Empty dictionary that we will add the amount of times a word shows up given a sense into
    senseTotals = defaultdict(int) # hashmap for the total amount each sense shows up. Kinda overkill but I'm getting used to working with hashmaps and thought this was easier.


    # Go through the sense to bag of words hashmap and count how many times each sense appears
    for sense in senseMap:
        for tokens in senseMap[sense]:
            senseTotals[sense] += 1

            # For every bag of words, add unseen tokens to the hashmap and add to the total amount of times it has been seen with the associated sense            
            for token in tokens:
                if token not in wordPerSense:
                    wordPerSense[token] = defaultdict(int)
                wordPerSense[token][sense] += 1

    # Build out the decision list for every unique word
    for token in wordPerSense:

        # grab the number of times the word has appeared in the product sense and the phone sense
        prodCount = wordPerSense[token]["product"]
        phoneCount = wordPerSense[token]["phone"]

        # the total amount of times the word appeared in the entire training data
        totCount = prodCount + phoneCount

        # If the word never appeared with a certain sense, it should weighted to indicate the other sense. I played with these numbers alot but settled on +-3 since 
        # most predictions that had a count for both were in the range of 1-5 and 3 felt middle of the road enough to not always favor but heavily weight. 
        if prodCount == 0:
            logScore = -3 
            match = "phone"

        elif phoneCount == 0: 
            logScore = 3
            match = "product"

        else:

            # If there are counts for both, do the monster math using the formula from the slides
            prodProb = prodCount / totCount
            phoneProb = phoneCount / totCount
            logScore = math.log2(prodProb / phoneProb)

            if(logScore > 0):
                match = "product"

            else:
                match = "phone"

        # Add a tuple of values to the decision list for every unique word we have
        decList.append((token, logScore, match))

    # print(senseTotals) for use in MFS

    # Use lambda function to grab logScore value from each tuple (shoutout 304 lambda functions are my new best friend)
    decList.sort(key=lambda x: abs(x[1]), reverse=True)    

    # Put results into our model file. 
    with open(modelFile, 'w', encoding ='utf-8') as file:
        for token, logScore, sense in decList:
            file.write(f"{token}\t{logScore}\t{sense}\n")

    return decList

def modelApply(dl, inFile):
    # Again this is just for removing these words at the end
    stopwords = {"line", "lines"}

    with open(inFile, 'r', encoding='utf-8') as file:
        instanceBlocks = re.findall(r"<instance.*?</instance>", file.read().lower(), re.DOTALL) # grab all the instances in the same way we did last time
    
    for instance in instanceBlocks:
        prediction = ""
        instanceID = re.search(r'id="(.*?)"', instance).group(1) # find the instance id for output formatting
        fullContext = re.search(r"<context>(.*?)</context>", instance, re.DOTALL).group(1) # make sense a string and not a list
        fullContext = re.sub(r"<head>", "", fullContext) # remove the head tag
        fullContext = re.sub(r"</head>", "", fullContext) # remove the head tag

        # grab tokens and filter stopwords like in the bow function
        tokens = set(re.findall(r"\w+", fullContext))
        tokens = tokens - stopwords

        # Go through the decision list, which has been sorted in order of strongest word association to weakest.
        # Once we hit a token in our decision list that is also in the test data context tokens, that is 
        # guarenteed to be the strongest association due to the way our decision list is sorted so we grab the sense and predict it. 
        for token, logScore, sense in dl:
            if token in tokens:
                prediction = sense
                break

        # prediction = "product" (used for MSF)

        # Print results in the same format as the key file
        print(f'<answer instance="{instanceID}" senseid="{prediction}"/>')

def main():

    # grab input files
    trainFile = sys.argv[1] 
    testFile = sys.argv[2]
    modelFile = sys.argv[3]

    # Do what must be done
    senseMap = bow(trainFile)
    trainedModel = dl(senseMap, modelFile)
    modelApply(trainedModel, testFile)

if __name__ == "__main__":
    main()
