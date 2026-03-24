# This file aims to check the output of our wsd model against a labeled version of the test data to indicate the accuracy of our model and create a confusion matrix 
# of the results. 

# Since both files are formatted the same, we use the same regex to grab the predicted sense from the output of wsd and the actual sense from the key file.
# Then we simply check the senses against each other to build out our confusion matrix, using product as the positive and phone as the negative as we did in the
# wsd file. 

# Analysis of the outputted confusion matrix and accuracy are documented in wsd but will be added again here for redundancy:

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

def main():

    # grab input files
    predFile = sys.argv[1] 
    keyFile = sys.argv[2]

    # parse input files
    with open(predFile, 'r', encoding = 'utf-16') as pF: # had to use utf-16 since that's how my pc formatted it idk if this will always work but I had to do it
        predSenses = re.findall(r'senseid="(.*?)"', pF.read().lower(), re.DOTALL) # we can use the same regex for both since they are formatted the same

    with open(keyFile, 'r', encoding='utf-8') as kF:
        keySenses = re.findall(r'senseid="(.*?)"', kF.read().lower(), re.DOTALL)# we can use the same regex for both since they are formatted the same

    # variables for total predictions, and the confusion matrix (true positive, true negative, etc.)
    tot = len(predSenses)
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Make the confusion matrix, with product as positive and phone as negative as was done in the wsd file
    for i in range(tot):

        if(predSenses[i] == "product" and keySenses[i] == "product"):
            tp += 1
        elif(predSenses[i] == "phone" and keySenses[i] == "phone"):
            tn += 1
        elif(predSenses[i] == "phone" and keySenses[i] == "product"):
            fn += 1
        else:
            fp += 1

    # Compute accuracy and print 
    accuracy = ((tp + tn) / tot) * 100

    # Format the confusion matrix (this took more tries than I would like to admit)
    print(f"Accuracy: {accuracy}")
    print("\t\tproduct\tphone")
    print(f"product\t\t{tp}\t{fn}")
    print(f"phone\t\t{fp}\t{tn}")
        
if __name__ == "__main__":
    main()