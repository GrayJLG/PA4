# This file aims to check the output of our ml model of choice against a labeled version of the test data to indicate the accuracy of our model and create a confusion matrix 
# of the results. 

# Since both files are formatted the same, we use the same regex to grab the predicted sense from the output of wsd and the actual sense from the key file.
# Then we simply check the senses against each other to build out our confusion matrix, using product as the positive and phone as the negative. 

# Analysis of the outputted confusion matrix and accuracy are documented in wsd-ml but will be added again here for redundancy:

# Below is a copy-paste of the analysis from wsd-ml of the accuracy and confusion matrices for each of the integrated ml models

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