from net import *


def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x.view(-1, 42))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(my_train_loader):
        data = Variable(data)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data , mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(my_train_loader.dataset),
                       5. * batch_idx / len(my_train_loader),
                       loss.data / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(my_train_loader.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     for data, _ in test_loader:
#         with torch.no_grad():
#             data = Variable(data)
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))


def get_my_train_set():
    with open('data/arrays/X_fraud_train.pkl', 'rb') as f:
        X_fraud = pickle.load(f)
    # dataset = []
    # for x in X_fraud:
    #     x = (x, 1)
    #     dataset.append(x)
    # print(sum(X_fraud))
    return X_fraud

# trainset = datasets.MNIST('../data', train=True, download=False, transform=transforms.ToTensor())
# print(trainset[0])
# train_loader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=128, shuffle=True)


my_train_set = get_my_train_set()
my_train_set = torch.FloatTensor(my_train_set)
my_train_loader = torch.utils.data.DataLoader(my_train_set, batch_size=995, shuffle=True)
# testset= datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
# test_loader = torch.utils.data.DataLoader(
#     testset,
#     batch_size=128, shuffle=True)

model = VAE()
reconstruction_function = nn.SmoothL1Loss()

reconstruction_function.size_average = False

optimizer = optim.Adam(model.parameters(), lr=1e-2)

if __name__ == '__main__':
    for epoch in range(1, 100):
        train(epoch)
        z = torch.randn(1, 3)
        xx = model.decode(z).detach()
        print(xx)
    #     test(epoch)

    print('===> Saving models...')
    state = {
        'state': model.state_dict()
    }

    torch.save(state, './model_file/VAE/autoencoder.t7')

    z = torch.randn(1, 3)
    xx = model.decode(z).detach()
    print(xx)
    pass
