import os
import numpy as np
import tensorflow as tf

def build_ff_model(x):
    # x.shape = [None, 784]
    # y.shape = [None]

    # weights & biases for feed-forward neural network
    W1 = tf.get_variable("W1", [784, 250], initializer=tf.glorot_normal_initializer())
    b1 = tf.get_variable("b1", [250], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [250, 10], initializer=tf.glorot_normal_initializer())
    b2 = tf.get_variable("b2", [10], initializer=tf.zeros_initializer())

    # computation
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.tanh(z1)

    z2 = tf.matmul(a1, W2) + b2
    nn_output = tf.nn.softmax(z2)

    return nn_output

def load_data(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    num = len(lines) - 1
    x = np.zeros(shape=(num,784))
    y = np.zeros(shape=(num))
    for i, line in enumerate(lines[1:]):
        items = line.strip().split(',')
        y[i] = int(items[0])
        for j, val in enumerate(items[1:]):
            x[i,j] = int(val)
    return x, y


def main():
    # placeholders
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y = tf.placeholder(tf.int32, [None], name="y")

    nn_output = build_ff_model(x)

    # loss function
    onehot = tf.one_hot(y, 10, dtype=tf.float32)
    loss = -1.0 * tf.reduce_mean( onehot * tf.log(nn_output) )

    # training op
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # evaluation
    correct_pred = tf.equal(tf.argmax(nn_output, 1, output_type=tf.int32), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # training
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # choose the device (GPU) here
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.


    intensities, labels = load_data('data/train.csv')
    intensities_test, labels_test = load_data('data/test.csv')

    batch_size = 512
    train_size = len(labels)

    # ---------------------- #
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)


    summary_op = tf.summary.merge_all()
    # ---------------------- #

    counter = 0

    with tf.Session(config=sess_config) as sess:
        # writer
        summary_writer = tf.summary.FileWriter('mnist_logs/', graph_def=sess.graph_def)

        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            for i in range(int(train_size/batch_size)):
                feed_dict = {x: intensities[i*batch_size:(i+1)*batch_size],
                             y: labels[i*batch_size:(i+1)*batch_size]}
                train_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                if i % 10 == 0:
                     print("batch: {} --- train_loss: {:.5f}".format(i, train_loss))

                     # write something
                     summary_str = sess.run(summary_op, feed_dict=feed_dict)
                     summary_writer.add_summary(summary_str, counter)
                     counter += 1


            print("################## EPOCH {} done ##################".format(epoch))
            feed_dict = {x: intensities_test, y: labels_test}
            [acc] = sess.run([accuracy], feed_dict=feed_dict)
            print("epoch: {} --- accuracy: {}".format(epoch, acc*100))


if __name__ == '__main__':
    main()
