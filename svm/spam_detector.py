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
    directory = 'spam'
    m = 500

    vocab_dick = getVocabDict()
    dick_size = len(vocab_dick)

    X = np.zeros((m, dick_size))
    Y = np.ones((m))[:, np.newaxis]
    
    for i in range(m):
        email_contents = codecs.open('../data/emails/{0}/{1:04d}.txt'.format(directory, i+1), 'r', encoding = 'utf 8', errors = 'ignore' ).read()
        email_contents = email2TokenList(email_contents)

        for word_idx in range(len(email_contents)):
            dick_index = vocab_dick.get(email_contents[word_idx])
            if (dick_index != None): X[i, dick_index-1] = 1

    print(X.shape)
    trained_svm = train(X, Y, 1, 0.1)

if __name__ == "__main__":
    main()
