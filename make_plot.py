import collections
import pickle
from wordcloud import WordCloud
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('log_word2vec/final_embeddings.pkl', 'rb') as f:
    final_embeddings = pickle.load(f)
with open('log_word2vec/reverse_dictionary.pkl', 'rb') as f:
    reverse_dictionary = pickle.load(f)
with open('log_word2vec/dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)


def plot_vocabulary():
    # count = []
    # count.extend(collections.Counter(vocabulary).most_common(12000 - 1))
    # print(count)
    with open('data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    result = " ".join(vocabulary)
    wc = WordCloud(
        # font_path='simhei.ttf',  # 字体路劲
        background_color='white',  # 背景颜色
        width=1000,
        height=600,
        max_font_size=50,  # 字体大小
        min_font_size=10,
        #   mask=plt.imread('xin.jpg'),  # 背景图片
        max_words=100,
        prefer_horizontal=1.0,
        repeat=False
    )
    wc.generate(result)
    wc.to_file('figures/wordcloud.png')
    plt.figure('wordcloud')  # 图片显示的名字
    plt.imshow(wc)
    plt.axis('off')  # 关闭坐标
    plt.show()
    
    pass


def plot_embeddings_3d():
    tsne = TSNE(
        perplexity=30, n_components=3, init='pca', n_iter=5000, method='exact')
    plot_only = 50
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection='3d')
    for i, label in enumerate(labels):
        x, y, z = low_dim_embs[i, :]
        ax.scatter(x, y, z)
        ax.text(x, y, z, label)
        # plt.annotate(
        #     label,
        #     xy=(x, y),
        #     xytext=(5, 2),
        #     textcoords='offset points',
        #     ha='right',
        #     va='bottom')
    plt.savefig('figures/embeddings_3d_70.png')
    plt.show()


def plot_embeddings_2d():
    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 2000
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #low_dim_embs = tsne.fit_transform(final_embeddings)
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 12))
    duration_x = []
    duration_y = []
    clock_x = []
    clock_y = []
    called_x = []
    called_y = []
    calling_x = []
    calling_y = []
    user_x = []
    user_y = []
    ring_x = []
    ring_y = []
    result_x = []
    result_y = []
    cause_x = []
    cause_y = []
    spam_x = []
    spam_y = []
    other_x = []
    other_y = []
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        if "duration" in label:
            duration_x.append(x)
            duration_y.append(y)
            color = 'black'
        elif "clock" in label:
            clock_x.append(x)
            clock_y.append(y)
            color = 'red'
        
        elif "called" in label:
            called_x.append(x)
            called_y.append(y)
            color = 'orange'
        elif "calling_hlr" in label:
            calling_x.append(x)
            calling_y.append(y)
            color = 'yellow'
        elif 'user' in label:
            user_x.append(x)
            user_y.append(y)
            color = 'green'
        elif 'ring' in label:
            ring_x.append(x)
            ring_y.append(y)
            color = 'blue'
        elif "result" in label:
            result_x.append(x)
            result_y.append(y)
            color = 'darkviolet'
        elif 'cause' in label:
            cause_x.append(x)
            cause_y.append(y)
            color = 'deeppink'
        elif "spam" in label:
            spam_x.append(x)
            spam_y.append(y)
            color = 'cyan'
        else:
            other_x.append(x)
            other_y.append(y)
    
    d1, = plt.plot(duration_x, duration_y, 'o')
    #plt.legend("duration")
    d2, = plt.plot(clock_x, clock_y, 'o')
    #plt.legend("clock")
    d3, = plt.plot(called_x, called_y, 'o')
    #plt.legend("called_hlr")
    d4, = plt.plot(calling_x, calling_y, 'o')
    #plt.legend("calling_hlr")
    #d5, = plt.plot(user_x, user_y, 'o')
   # plt.legend("user")
    d6, = plt.plot(ring_x, ring_y, 'o')
    #plt.legend("ring_time")
    d7, = plt.plot(result_x, result_y, 'o')
   # plt.legend("result")
    d8, = plt.plot(cause_x, cause_y, 'o')
   # plt.legend("cause")
    d9, = plt.plot(spam_x, spam_y, 'o')
   # plt.legend("spam_label")
    d10, = plt.plot(other_x, other_y, 'o')
    #plt.legend("other field")
    plt.legend(handles=[d1, d2, d3, d4,  d6, d7, d8, d9, d10],
               labels=["duration", "clock", "called_hlr", "calling_hlr", "ring_time", "result", "cause",
                       "spam_label", "other field"])
    
    # plt.annotate(
    #     label,
    #     xy=(x, y),
    #     xytext=(5, 2),
    #     textcoords='offset points',
    #     ha='right',
    #     va='bottom')
    
    plt.savefig('figures/embeddings_2d_2000_no_user.png')
    plt.show()


if __name__ == '__main__':
    # plot_vocabulary()
    plot_embeddings_2d()
    pass
