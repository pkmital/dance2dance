import numpy as np
import tensorflow as tf
import seq2seq


def batch_generator(data,
                    sequence_length,
                    batch_size=50):
    idxs = np.random.permutation(np.arange(len(data) - sequence_length))
    n_batches = len(idxs) // batch_size
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        batch = [data[i:i + sequence_length, :] for i in this_idxs]
        yield np.array(batch, dtype=np.float32)


def train(data,
          n_epochs=1000,
          batch_size=100,
          sequence_length=240,
          **kwargs):

    n_features = data.shape[1]
    config = tf.ConfigProto(allow_gpu_growth=True)
    sess = tf.Session(config=config)

    net = seq2seq.create_model(
        batch_size=batch_size,
        n_features=n_features,
        sequence_length=sequence_length)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    current_learning_rate = 0.01
    for epoch_i in range(n_epochs):
        total = 0
        for it_i, batch in enumerate(batch_generator(
                data, sequence_length=sequence_length, batch_size=batch_size)):
            if it_i % 1000 == 0:
                current_learning_rate = max(0.0001,
                                            current_learning_rate * 0.99)
                print(it_i)
            loss = sess.run(
                [net['loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['keep_prob']: 0.8,
                    net['data']: batch
                })[0]
            total = total + loss
            print('{}: {}'.format(it_i, total / (it_i + 1)), end='\r')
        # End of epoch, save
        print('epoch {}: {}'.format(epoch_i, total / it_i))
        saver.save(sess, './seq2seq.ckpt', global_step=it_i)

    sess.close()


def run_euler():
    data = np.load('euler.npy')
    data = data.reshape([data.shape[0], -1])
    train(data)
