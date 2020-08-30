import numpy as np
from keras.datasets import mnist


def relu(x):
    return (x > 0) * x


def relu2deriv(x):
    return x >= 0


def print_result(iteration, error, correct_cnt, test_error, test_correct_cnt, number_to_test):
    print('Iteration: {iteration}; Train-Err: {train_error}; Train-Acc: {train_acc}; '
          'Test-Err: {test_error}; Test-Acc: {test_acc}'.format(
            iteration=iteration,
            test_error=test_error / number_to_test,
            test_acc=test_correct_cnt / number_to_test,
            train_error=error / number_to_test,
            train_acc=correct_cnt / number_to_test
    ))


def stochastic_gradient_descent(images, labels, weights_0_1, weights_1_2, iterations, number_to_test,
                                images_test, labels_test):
    print('stochastic_gradient_descent')
    alpha = 0.005
    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(number_to_test):
            layer_0 = images[i:i + 1]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = np.dot(layer_1, weights_1_2)
            error += np.sum((layer_2 - labels[i:i + 1]) ** 2)
            correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i + 1]))

            layer_2_delta = (labels[i:i + 1] - layer_2)
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        if j % 10 == 0 or j == iterations - 1:
            test_error, test_correct_cnt = (0, 0)
            for i in range(number_to_test):
                layer_0 = images_test[i:i + 1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)
                test_error += np.sum((layer_2 - labels_test[i:i + 1]) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(labels_test[i:i + 1]))

            print_result(iteration=j, error=error, correct_cnt=correct_cnt, test_error=test_error,
                         test_correct_cnt=test_correct_cnt, number_to_test=number_to_test)


def batch_gradient_descent(images, labels, weights_0_1, weights_1_2, iterations, number_to_test,
                           images_test, labels_test):
    print('batch_gradient_descent')
    alpha = 0.001
    batch_size = 100
    for j in range(iterations):
        error = 0
        correct_cnt = 0
        for i in range(int(number_to_test/batch_size)):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            layer_0 = images[batch_start:batch_end]
            layer_1 = relu(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = np.dot(layer_1, weights_1_2)
            error += np.sum((layer_2 - labels[batch_start:batch_end]) ** 2)

            for k in range(batch_size):
                correct_cnt += int(np.argmax(layer_2[k:k + 1]) == np.argmax(
                    labels[batch_start + k:batch_start + k + 1]
                ))
                layer_2_delta = (labels[batch_start:batch_end] - layer_2)/batch_size
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
                layer_1_delta *= dropout_mask

                weights_1_2 += layer_1.T.dot(layer_2_delta) * alpha
                weights_0_1 += layer_0.T.dot(layer_1_delta) * alpha

        if j % 10 == 0:
            test_error = 0
            test_correct_cnt = 0
            for i in range(number_to_test):
                layer_0 = images_test[i:i+1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                test_error += np.sum((layer_2 - labels_test[i:i+1]) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(labels_test[i:i+1]))

            print_result(iteration=j, error=error, correct_cnt=correct_cnt, test_error=test_error,
                         test_correct_cnt=test_correct_cnt, number_to_test=number_to_test)



if '__main__' == __name__:
    # x_train, x_test - images
    # y_train, y_test - labels
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f'Training data shape: {x_train.shape}')
    print(f'Test data shape: {x_test.shape}')

    number_to_test = 1000
    image_size = 28 * 28
    images = x_train[0:number_to_test].reshape(number_to_test, image_size) / 255

    num_labels = 10
    # prepare number vector (one_hot_encoded_vector)
    labels = np.zeros((number_to_test, num_labels))
    for i in range(number_to_test):
        labels[i][y_train[i]] = 1

    images_test = x_test[0:number_to_test].reshape(number_to_test, image_size) / 255
    labels_test = np.zeros((number_to_test, num_labels))
    for i in range(number_to_test):
        labels_test[i][y_test[i]] = 1

    np.random.seed(1)

    iterations = 300
    hidden_size = 100

    weights_0_1 = 0.2 * np.random.random((image_size, hidden_size)) - 0.1
    weights_1_2 = 0.2 * np.random.random((hidden_size, num_labels)) - 0.1

    stochastic_gradient_descent(images=images, labels=labels, weights_0_1=weights_0_1, weights_1_2=weights_1_2,
                           iterations=iterations, number_to_test=number_to_test, images_test=images_test,
                           labels_test=labels_test)

    batch_gradient_descent(images=images, labels=labels, weights_0_1=weights_0_1, weights_1_2=weights_1_2,
                           iterations=iterations, number_to_test=number_to_test, images_test=images_test,
                           labels_test=labels_test)
