import sys
import numpy as np

with open('reviews.txt') as f:
    raw_reviews = f.readlines()

with open('labels.txt') as f:
    raw_labels = f.read().splitlines()


tokens = list(map(lambda x: set(x.split()), raw_reviews))

vocabulary = set()

for sent in tokens:
    for word in sent:
        vocabulary.add(word)

word2index = {}
for i, word in enumerate(vocabulary):
    word2index[word] = i

input_dataset = list()
test = list()
for sent in tokens:
    sent_indices = list()
    for word in sent:
        try:
            sent_indices.append(word2index[word])
        except:
            pass

    input_dataset.append(list(set(sent_indices)))

target_dataset = list()
for label in raw_labels:
    target_dataset.append(int('positive' == label))


def sigmoid(x):
    return 1/(1 + np.exp(-x))


alpha = 0.01
iterations = 2
hidden_size = 100

weights_0_1 = 0.2 * np.random.random((len(vocabulary), hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size, 1)) - 0.1
correct = 0
total = 0
number_to_test = 24000

for iter in range(iterations):
    for i in range(number_to_test):
        x = input_dataset[i]
        y = target_dataset[i]

        layer_1 = sigmoid(np.sum(weights_0_1, axis=x))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)
