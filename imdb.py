import sys
import numpy as np
import math
from collections import Counter

np.random.seed(1)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


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

        layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
        layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

        layer_2_delta = layer_2 - y
        layer_1_delta = layer_2_delta.dot(weights_1_2.T)

        weights_0_1[x] -= layer_1_delta * alpha
        weights_1_2 -= np.outer(layer_1, layer_2_delta) * alpha

        if (np.abs(layer_2_delta)) < 0.5:
            correct += 1
        total += 1

        if i % 10 == 9:
            progress = str(i / float(len(input_dataset)))
            training_accuracy = correct / total
            sys.stdout.write(f'\rIter: {iter}; progress: {progress}%; training accuracy: {training_accuracy}%')

    print()

correct = 0
total = 0

for i in range(number_to_test, len(input_dataset)):
    x = input_dataset[i]
    y = target_dataset[i]

    layer_1 = sigmoid(np.sum(weights_0_1[x], axis=0))
    layer_2 = sigmoid(np.dot(layer_1, weights_1_2))

    if np.abs(layer_2 - y) < 0.5:
        correct += 1
    total += 1

training_accuracy = correct / total
print(f'test accuracy: {training_accuracy}')


def similar(target):
    target_index = word2index[target]
    scores = Counter()
    for word, index in word2index.items():
        raw_difference = weights_0_1[index] - weights_0_1[target_index]
        squared_difference = raw_difference ** 2
        scores[word] = - math.sqrt(sum(squared_difference))

    return scores.most_common(10)

print(*similar('beautiful'), sep='\n')