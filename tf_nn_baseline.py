import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt


def makedataset():
    with open('log_word2vec/final_embeddings.pkl', 'rb') as f:
        final_embeddings = pickle.load(f)
    with open('log_word2vec/reverse_dictionary.pkl', 'rb') as f:
        reverse_dictionary = pickle.load(f)
    with open('log_word2vec/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    with open('data/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
        
    # print(reverse_dictionary) # {0: 'UNK', 1: 'result_0', 2: 'call_duration_0', 3: 'ring_time_0', 4: 'is_spam', 5: 'cause_08F90', 6: 'is_not_spam', 7: 'cause_08090', 8: 'ring_time_30', 9: 'calling_hlr_01', 10: 'called_hlr_0579',
    # print(dictionary) # {'UNK': 0, 'result_0': 1, 'call_duration_0': 2, 'ring_time_0': 3, 'is_spam': 4, 'cause_08F90': 5, 'is_not_spam': 6, 'cause_08090': 7, 'ring_time_30': 8, 'calling_hlr_01': 9, 'called_hlr_0579': 10, 'called_hlr_0574': 11,
    print(vocabulary[:9])
    X = []
    y = []
    for i in range(len(vocabulary)//9):
        calling_hlr = vocabulary[i*9+0]
        called_hlr = vocabulary[i*9+1]
        user_id = vocabulary[i*9+2]
        clock = vocabulary[i*9+3]
        spam = vocabulary[i*9+4]
        ring = vocabulary[i*9+5]
        duration = vocabulary[i*9+6]
        result = vocabulary[i*9+7]
        cause = vocabulary[i*9+8]
        
        tmp = []
        index = dictionary.get(calling_hlr, 0)
        calling_hlr = final_embeddings[index]
        index = dictionary.get(called_hlr, 0)
        called_hlr = final_embeddings[index]
        index = dictionary.get(user_id, 0)
        user_id = final_embeddings[index]
        index = dictionary.get(clock, 0)
        clock = final_embeddings[index]
        index = dictionary.get(ring, 0)
        ring = final_embeddings[index]
        index = dictionary.get(duration, 0)
        duration = final_embeddings[index]
        index = dictionary.get(result, 0)
        result = final_embeddings[index]
        index = dictionary.get(cause, 0)
        cause = final_embeddings[index]
        
        tmp.extend(calling_hlr)
        tmp.extend(called_hlr)
        tmp.extend(user_id)
        tmp.extend(clock)
        tmp.extend(ring)
        tmp.extend(duration)
        tmp.extend(result)
        tmp.extend(cause)
        
        X.append(np.array(tmp))
        if spam == "is_not_spam":
            y.append(0)
        else:
            y.append(1)
    print(len(X), len(y), sum(y))
    with open("data/embeddings/X.pkl", 'wb') as f:
        pickle.dump(X, f)
    with open("data/embeddings/y.pkl", 'wb') as f:
        pickle.dump(y, f)
    
        
        
    
    
        

def train():
    with open('data/embeddings/X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('data/embeddings/y.pkl', 'rb') as f:
        y = pickle.load(f)
        
    X = np.array(X)
    y = np.array(y)
    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
    
    model = keras.Sequential([
        keras.layers.Dense(input_shape=[len(train_data[0])], activation=tf.nn.sigmoid, units=9),
        #keras.layers.Dense(30, activation=tf.nn.relu),
        #keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    
    print(model.summary())
    
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    x_val = train_data[:1000]
    partial_x_train = train_data[1000:]
    
    y_val = train_labels[:1000]
    partial_y_train = train_labels[1000:]
    
    history = model.fit(train_data,
                        train_labels,
                        epochs=40,
                        batch_size=32,
                        validation_data=(test_data, test_labels),
                        verbose=1)
    results = model.evaluate(test_data, test_labels)
    print(results)

    history_dict = history.history

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/train_loss.png')
    plt.show()
    
    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('figures/train_acc.png')
    plt.show()


if __name__ == '__main__':
    print(tf.__version__)
    # makedataset()
    train()
    






