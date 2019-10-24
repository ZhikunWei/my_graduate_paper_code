import numpy as np

from net import *


def get_my_train_set():
    with open('data/arrays/X_fraud.pkl', 'rb') as f:
        X_fraud = pickle.load(f)
    return X_fraud


if __name__ == '__main__':
    model = VAE()
    checkpoint = torch.load('./model_file/VAE/autoencoder.t7')
    model.load_state_dict(checkpoint['state'])
    X_fraud = get_my_train_set()
    X_fraud = torch.FloatTensor(X_fraud)
    X_fraud_gen = []
    for x in X_fraud:
        mu, logvar = model.encode(x.view(-1, 42))

        z = model.reparametrize(mu, logvar)
        print(z)
        x_new = model.decode(z).detach().numpy()
        X_fraud_gen.append(x_new)

    print(sum(X_fraud))
    print(sum(X_fraud_gen))
    pass

