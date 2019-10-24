import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def readFromFile(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def extractFeature():
    X_doc_train = readFromFile('data/lda/X_train.pkl')
    X_doc_test = readFromFile('data/lda/X_test.pkl')
    tf_vectorizer = CountVectorizer()
    tf = tf_vectorizer.fit_transform(X_doc_train)
    lda = LatentDirichletAllocation(n_components=50)
    X_feature_train = lda.fit_transform(tf)

    tf = tf_vectorizer.transform(X_doc_test)
    X_feature_test = lda.transform(tf)
    with open('model_file/lda/tf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tf_vectorizer, f)
    with open('model_file/lda/lda.pkl', 'wb') as f:
        pickle.dump(lda, f)
    with open('data/lda/X_feature_train.pkl', 'wb') as f:
        pickle.dump(X_feature_train, f)
    with open('data/lda/X_feature_test.pkl', 'wb') as f:
        pickle.dump(X_feature_test, f)


if __name__ == '__main__':
    extractFeature()
    pass
