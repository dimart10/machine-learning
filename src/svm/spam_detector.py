import numpy as np
import matplotlib.pyplot as plt
from  scipy.io import loadmat
from sklearn.svm import SVC
from svm import *
from process_email import *
from get_vocab_dict import *
import codecs

def main():
    # DATA PREPROCESSING
    vocab_dick = getVocabDict()
    dick_size = len(vocab_dick)

    validationPercent = 0.3

    # SPAM
    directorySpam = 'spam'
    mSpam = 500

    X_spam = np.zeros(((int)(mSpam * (1-validationPercent)), dick_size))
    Y_spam = np.ones(((int)(mSpam * (1-validationPercent))))[:, np.newaxis]

    X_spam_val = np.zeros(((int)(mSpam * validationPercent), dick_size))
    Y_spam_val = np.ones((int)(mSpam * validationPercent))[:, np.newaxis]

    for i in range(mSpam):
        email_contents = codecs.open('../data/emails/{0}/{1:04d}.txt'.format(directorySpam, i+1), 'r', encoding = 'utf 8', errors = 'ignore' ).read()
        email_contents = email2TokenList(email_contents)

        val = i >= mSpam * (1-validationPercent)
        currentX = X_spam if not val else X_spam_val
        for word_idx in range(len(email_contents)):
            dick_index = vocab_dick.get(email_contents[word_idx])
            if (dick_index != None):
                currentX[i if not val else (int)(i - mSpam * (1-validationPercent)), dick_index-1] = 1


    # EASY HAM
    directoryEasy = 'easy_ham'
    mEasy = 500

    X_easy = np.zeros(((int)(mEasy * (1-validationPercent)), dick_size))
    Y_easy = np.zeros(((int)(mEasy * (1-validationPercent))))[:, np.newaxis]

    X_easy_val = np.zeros(((int)(mEasy * validationPercent), dick_size))
    Y_easy_val = np.zeros((int)(mEasy * validationPercent))[:, np.newaxis]

    for i in range(mEasy):
        email_contents = codecs.open('../data/emails/{0}/{1:04d}.txt'.format(directoryEasy, i+1), 'r', encoding = 'utf 8', errors = 'ignore' ).read()
        email_contents = email2TokenList(email_contents)

        val = i >= mEasy * (1-validationPercent)
        currentX = X_easy if not val else X_easy_val
        for word_idx in range(len(email_contents)):
            dick_index = vocab_dick.get(email_contents[word_idx])
            if (dick_index != None):
                currentX[i if not val else (int)(i - mEasy * (1-validationPercent)), dick_index-1] = 1


    # HARD HAM
    directoryhard = 'hard_ham'
    mhard = 250

    X_hard = np.zeros(((int)(mhard * (1-validationPercent)), dick_size))
    Y_hard = np.zeros(((int)(mhard * (1-validationPercent))))[:, np.newaxis]

    X_hard_val = np.zeros(((int)(mhard * validationPercent), dick_size))
    Y_hard_val = np.zeros((int)(mhard * validationPercent))[:, np.newaxis]

    for i in range(mhard):
        email_contents = codecs.open('../data/emails/{0}/{1:04d}.txt'.format(directoryhard, i+1), 'r', encoding = 'utf 8', errors = 'ignore' ).read()
        email_contents = email2TokenList(email_contents)

        val = i >= mhard * (1-validationPercent)
        currentX = X_hard if not val else X_hard_val
        for word_idx in range(len(email_contents)):
            dick_index = vocab_dick.get(email_contents[word_idx])
            if (dick_index != None):
                currentX[i if not val else (int)(i - mhard * (1-validationPercent)), dick_index-1] = 1


    # Mix spam with non spam
    X = np.vstack((X_spam, X_easy))
    X = np.vstack((X, X_hard))
    Y = np.vstack((Y_spam, Y_easy))
    Y = np.vstack((Y, Y_hard))

    X_val = np.vstack((X_spam_val, X_easy_val))
    X_val = np.vstack((X_val, X_hard_val))
    Y_val = np.vstack((Y_spam_val, Y_easy_val))
    Y_val = np.vstack((Y_val, Y_hard_val))

    # Finally, train the SVM and test the results
    # trained_svm = train(X, Y, 1, 0.1)
    # success_percentage = test(trained_svm, X_val, Y_val)

    possible_values = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
    bestSvmResults = findBestSVM(X, Y, X_val, Y_val, possible_values, possible_values)
    #success_percentage = bestSvmResults[-1]

    #print("Success percentage: ", success_percentage * 100, "%")

if __name__ == "__main__":
    main()
