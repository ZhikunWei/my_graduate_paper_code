import pickle
import numpy as np
from sklearn import svm

def train_svm():
    clf = svm.SVC(gamma='scale')
    with open('data/lda/X_feature_train.pkl', 'rb') as f:
        X_feature_train = pickle.load(f)
    with open('data/lda/y_train.pkl', 'rb') as f:
        y_feature_train = pickle.load(f)
    with open('data/lda/X_feature_test.pkl', 'rb') as f:
        X_feature_test = pickle.load(f)
    clf.fit(X_feature_train, y_feature_train)

    y_predict = clf.predict(X_feature_test)

    with open('model_file/lda/svm.pkl', 'wb') as f:
        pickle.dump(clf, f)


def test_svm():
    with open('model_file/lda/svm.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('data/lda/X_feature_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/lda/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)

    #y_predict = clf.predict(X_test)
    #print(y_predict.sum())
    print(np.sum(y_test), len(y_test))


if __name__ == '__main__':
    # train_svm()
    test_svm()
    pass
