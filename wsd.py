# The purpose of this program is to implement a decision list that performs word sense disambiguation using bag of words feature representation. If that last sentence was greek,
# we can define each:

# Word sense disambiguation is the process of using the context surrounding a word to try and determine its true meaning. In this program we try
# to determine if the word "line(s)" is referring to a phone line or a product line using the context of the surrounding sentence(s).

# bag of words feature representation is the process of associating all of the other words surrounding the disambigous word with how the word was meant to be used. We can 
# purposefully remove words that don't add much context using a "stoplist" which you can reference in the "bow" function below. 

# putting it all together, a decision list captures a surrounding context word, a "score" of that word's association with a certian meaning of the disambiguous word, 
# and a "sense" which indicates how the disamiguous word was meant to be used. This can be seen in the "dl" function below

# Here is some example input and output, first by just running the wsd.py program doing:

# python3 wsd.py line-train.txt line-test.txt my-model.txt where line-train.txt is the training data, line-test.txt is the testing data, and my-model.txt will contain the final
# decision list

# Output will be: 
# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="phone"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="phone"/>
# <answer instance="line-n.w8_119:16927:" senseid="product"/>
# ...
# where each line indicates the context instance in the TEST data and your model's prediction for the sense of that instance

# running wsd.py and scorer.py in tandem will give you insight into the accuracy of your model and the confusion matrix, which can be done by doing:
# python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
# python3 scorer.py my-line-answers.txt line-key.txt  
# where my-line-answers.txt simply captures the same output as running only wsd.py from above and line-key.txt is the labeled test data where the correct sense is indicated.

# Output will be:
# Accuracy: 81.74603174603175
#                 product phone
# product         33      21
# phone           2       70

# Now how do we actually do this in the program?

# In order to determine this, we first train a model on labeled data that has many examples of our target word in different contexts. Crucially, this data tells us what sense
# the word in the sentence is being used in so we can establish an inital bag of words to sense association.

# After creating a hashmap containg an association between a bag of words and a sense from the training data, we create a decision list by doing the following:
# 1. gather a total for how many times each sense appeared in the training data
# 2. gather a total for each token of how many times it appeared with a specific sense (how many times with the phone sense, and how many times with the product sense)
# 3. if a token only appeared with one sense, give it an arbitray weight that makes that sense more favorable
# 4. if a token appears with both senses, use a logrithmic formula to determine if the word is more associated with the product sense or the phone sense
# 5. add a tuple to the decision list containing the token, the output from the logrithmic formula, and the sense the word is associated with
# 6. sort the decision list in descending order, since a higher logrithmic output indicates a stronger association with a sense. 

# Now that we have our decision list, we can apply it to the test data by doing the following:
# 1. for each contextual instance, use the same method and gather a bag of words.
# 2. for each bag of words captured, start with the strongest association in the decision list and check if that token appears in the bag of words. Since our decision list
# is in descending order we can simply start at index 0 and iterate until we find a match.
# 3. Once a matching association is found, check what sense was indicated in that association and use it as our prediction. 

# After the testing data has been fully iterated through, scorer.py compares our output to the key data to determine our accuracy and gives a confusion matrix. Information
# on the exact process for that can be found in the scorer.py file but the decision list, accuracy, and confusion matrix will be assessed here:

# My decision list is a three tuple containing (token, log score, sense) sorted in descending order by log score. 
# This is done so that when classifying the testing data we can start with the token with the highest log score, check if it is in the test data bag of words, and continue. 
# This guarentees we always grab the sense with the highest logscore for that token. 

# My accuracy is 81.74603174603175 which outperforms MFS baseline of 42.857142857142854 by ~39%

# My confusion matrix is:
#                 product phone
# product         33      21
# phone           2       70

# Which indicates a bias towards phone and difficulty distiguishing product

# This program was written by Jacob Gray for CMSC 437 taught by Dr. Bridget McInness at Virginia Commonwealth University
# Last Update: 3/18/2026

import sys # for parsing user input
import re # for regex
import math # for log functions
from collections import defaultdict # for bag of words

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
