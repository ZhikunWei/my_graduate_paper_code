import pickle

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import net


def get_my_train_set():
    with open('data/arrays/X_fraud.pkl', 'rb') as f:
        X_fraud = pickle.load(f)
    print(X_fraud[0:2])
    return X_fraud[0:200]


def get_normal_user():
    with open('data/arrays/X_normal.pkl', 'rb') as f:
        X_normal = pickle.load(f)
    print(sum(X_normal))
    return X_normal[0:200]


if __name__ == '__main__':
    model = net.VAE()
    checkpoint = torch.load('./model_file/VAE/autoencoder.t7')
    model.load_state_dict(checkpoint['state'])

    X_fraud = get_my_train_set()
    #X_fraud = get_normal_user()
    X_fraud = torch.FloatTensor(X_fraud)
    mu, log = model.encode(X_fraud.view(-1, 42))

    vec = model.reparametrize(mu, log)

    vec = vec.detach().numpy()

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter3D(vec[:, 0], vec[:, 1], vec[:, 2], cmap='Blue')
    plt.show()


