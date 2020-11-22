import numpy as np
from keras.datasets import mnist


def relu(x):
    return (x > 0) * x


def relu2deriv(x):
    return x >= 0


def tanh(x):
    return np.tanh(x)


def tanh2deriv(x):
    return 1 - x ** 2


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)


def print_result(iteration, error, correct_cnt, test_error, test_correct_cnt, number_to_test):
    print('Iteration: {iteration}; Train-Err: {train_error}; Train-Acc: {train_acc}; '
          'Test-Err: {test_error}; Test-Acc: {test_acc}'.format(
        iteration=iteration,
        test_error=test_error / number_to_test if test_error else '-',
        test_acc=test_correct_cnt / number_to_test,
        train_error=error / number_to_test if error else '-',
        train_acc=correct_cnt / number_to_test
    ))


def stochastic_gradient_descent(images, labels, weights_0_1, weights_1_2, iterations, number_to_test,
                                images_test, labels_test):
    print('stochastic_gradient_descent')
    alpha = 0.005
    for j in range(iterations):
        error = 0
        correct_cnt = 0
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
        for i in range(int(number_to_test / batch_size)):
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
                layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
                layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
                layer_1_delta *= dropout_mask

                weights_1_2 += layer_1.T.dot(layer_2_delta) * alpha
                weights_0_1 += layer_0.T.dot(layer_1_delta) * alpha

        if j % 10 == 0:
            test_error = 0
            test_correct_cnt = 0
            for i in range(number_to_test):
                layer_0 = images_test[i:i + 1]
                layer_1 = relu(np.dot(layer_0, weights_0_1))
                layer_2 = np.dot(layer_1, weights_1_2)

                test_error += np.sum((layer_2 - labels_test[i:i + 1]) ** 2)
                test_correct_cnt += int(np.argmax(layer_2) == np.argmax(labels_test[i:i + 1]))

            print_result(iteration=j, error=error, correct_cnt=correct_cnt, test_error=test_error,
                         test_correct_cnt=test_correct_cnt, number_to_test=number_to_test)


def activation_func_batch_gradient_descent(images, labels, weights_0_1, weights_1_2, iterations, number_to_test,
                                           images_test, labels_test):
    print('activation_funct_batch_gradient_descent')
    alpha = 2
    batch_size = 100
    for j in range(iterations):
        correct_cnt = 0
        for i in range(int(number_to_test / batch_size)):
            batch_start = i * batch_size
            batch_end = (i + 1) * batch_size
            layer_0 = images[batch_start:batch_end]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = softmax(np.dot(layer_1, weights_1_2))

            for k in range(batch_size):
                correct_cnt += int(
                    np.argmax(layer_2[k:k + 1]) == np.argmax(labels[batch_start + k:batch_start + k + 1]))

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        test_correct_cnt = 0
        for i in range(number_to_test):
            layer_0 = images_test[i:i + 1]
            layer_1 = tanh(np.dot(layer_0, weights_0_1))
            layer_2 = np.dot(layer_1, weights_1_2)
            test_correct_cnt += int(np.argmax(layer_2) == np.argmax(labels_test[i:i + 1]))
        if j % 10 == 0:
            print_result(iteration=j, error=None, correct_cnt=correct_cnt, test_error=None,
                         test_correct_cnt=test_correct_cnt, number_to_test=number_to_test)


def convolutional_neural_network(images, labels, num_labels, iterations, number_to_test,
                                 images_test, labels_test):
    print('convolutional_neural_network')
    np.random.seed(1)
    alpha = 2
    batch_size = 128
    input_rows = 28
    input_cols = 28

    kernel_rows = 3
    kernel_cols = 3
    num_kernels = 16

    hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels

    kernels = 0.02 * np.random.random((kernel_rows * kernel_cols, num_kernels)) - 0.01
    weights_1_2 = 0.2 * np.random.random((hidden_size,
                                          num_labels)) - 0.1
    for j in range(iterations):
        correct_cnt = 0
        for i in range(int(number_to_test / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            layer_0 = images[batch_start:batch_end]
            layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
            layer_0.shape

            sects = list()
            for row_start in range(layer_0.shape[1] - kernel_rows):
                for col_start in range(layer_0.shape[2] - kernel_cols):
                    sect = get_image_section(layer_0,
                                             row_start,
                                             row_start + kernel_rows,
                                             col_start,
                                             col_start + kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=1)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)

            kernel_output = flattened_input.dot(kernels)
            layer_1 = tanh(kernel_output.reshape(es[0], -1))
            dropout_mask = np.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = softmax(np.dot(layer_1, weights_1_2))

            for k in range(batch_size):
                labelset = labels[batch_start + k:batch_start + k + 1]
                _inc = int(np.argmax(layer_2[k:k + 1]) ==
                           np.argmax(labelset))
                correct_cnt += _inc

            layer_2_delta = (labels[batch_start:batch_end] - layer_2) \
                            / (batch_size * layer_2.shape[0])
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * \
                            tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
            k_update = flattened_input.T.dot(l1d_reshape)
            kernels -= alpha * k_update

        test_correct_cnt = 0

        for i in range(number_to_test):
            layer_0 = images_test[i:i + 1]
            layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
            layer_0.shape

            sects = list()
            for row_start in range(layer_0.shape[1] - kernel_rows):
                for col_start in range(layer_0.shape[2] - kernel_cols):
                    sect = get_image_section(layer_0,
                                             row_start,
                                             row_start + kernel_rows,
                                             col_start,
                                             col_start + kernel_cols)
                    sects.append(sect)

            expanded_input = np.concatenate(sects, axis=1)
            es = expanded_input.shape
            flattened_input = expanded_input.reshape(es[0] * es[1], -1)

            kernel_output = flattened_input.dot(kernels)
            layer_1 = tanh(kernel_output.reshape(es[0], -1))
            layer_2 = np.dot(layer_1, weights_1_2)

            test_correct_cnt += int(np.argmax(layer_2) ==
                                    np.argmax(labels_test[i:i + 1]))
        if j % 10 == 0:
            print_result(j, None, correct_cnt, None, test_correct_cnt, number_to_test)


def get_image_section(layer, row_form, row_to, col_from, col_to):
    section = layer[:, row_form:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_form, col_to - col_from)


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

    weights_0_1_tmp = np.random.random((image_size, hidden_size))
    weights_1_2_tmp = np.random.random((hidden_size, num_labels))

    weights_0_1 = 0.2 * weights_0_1_tmp - 0.1
    weights_1_2 = 0.2 * weights_1_2_tmp - 0.1

    stochastic_gradient_descent(images=images, labels=labels, weights_0_1=weights_0_1, weights_1_2=weights_1_2,
                                iterations=iterations, number_to_test=number_to_test, images_test=images_test,
                                labels_test=labels_test)

    batch_gradient_descent(images=images, labels=labels, weights_0_1=weights_0_1, weights_1_2=weights_1_2,
                           iterations=iterations, number_to_test=number_to_test, images_test=images_test,
                           labels_test=labels_test)

    weights_0_1_gradient_descent = 0.02 * weights_0_1_tmp - 0.01
    activation_func_batch_gradient_descent(images=images, labels=labels, weights_0_1=weights_0_1_gradient_descent,
                                           weights_1_2=weights_1_2,
                                           iterations=iterations, number_to_test=number_to_test,
                                           images_test=images_test,
                                           labels_test=labels_test)

    convolutional_neural_network(images=images, labels=labels, num_labels=num_labels, iterations=iterations,
                                 number_to_test=number_to_test,
                                 images_test=images_test, labels_test=labels_test)
