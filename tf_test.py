import os
import pickle


with open('log_word2vec/final_embeddings.pkl', 'rb') as f:
    final_embeddings = pickle.load(f)
with open('log_word2vec/reverse_dictionary.pkl', 'rb') as f:
    reverse_dictionary = pickle.load(f)
with open('log_word2vec/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)


# Step 6: Visualize the embeddings.


# pylint: disable=missing-docstring
# Function to draw visualization of distance between embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')
    plt.savefig(filename)
    plt.show()
    
    
    # pylint: disable=g-import-not-at-top


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(
    perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
plot_only = 50
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels, 'log_word2vec/tsne.png')
print("over?")


