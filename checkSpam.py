import pickle


def checkspam():
    with open('data/arrays/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/arrays/y.pkl', 'rb') as f:
        y = pickle.load(f)
    spamsum = X[0]
    for x, yy in zip(X, y):
        if yy is 1:
            print(x[:12], x[12:24], x[24:28], x[28:34], x[34:37], x[37:42])


if __name__ == '__main__':
    checkspam()
