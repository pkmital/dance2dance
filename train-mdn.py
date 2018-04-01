import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seq2seq


def plot(data):
    framerate = 60
    points = 2000
    skip = int(len(data) / points)
    duration = len(data) / float(framerate)
    data_skipped = data.transpose(1, 0, 2)[:, ::skip, :]
    plt.figure(figsize=(16, 10), facecolor='white')
    for i, joint in enumerate(data_skipped):
        plt.gca().set_prop_cycle(None)
        plt.plot(np.linspace(0, duration, len(joint)), i + joint, lw=1)
    plt.ylim([0, data_skipped.shape[0]])
    plt.xlim([0, duration])
    plt.ylabel('Joint')
    plt.xlabel('Seconds')
    plt.show()


def batch_generator(data, sequence_length, batch_size=50):
    idxs = np.random.permutation(np.arange(len(data) - sequence_length * 2))
    n_batches = len(idxs) // (batch_size * sequence_length)
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        source = [data[i:i + sequence_length, :] for i in this_idxs]
        target = [
            data[i + sequence_length:i + sequence_length * 2, :]
            for i in this_idxs
        ]
        yield np.array(
            source, dtype=np.float32), np.array(
                target, dtype=np.float32)


def train(data, mean_data, std_data, n_epochs=1000, batch_size=100, sequence_length=240, **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = seq2seq.create_model(
        batch_size=batch_size,
        sequence_length=sequence_length,
        **kwargs)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    current_learning_rate = 0.01
    for epoch_i in range(n_epochs):
        total, total_mse, total_mdn = 0.0, 0.0, 0.0
        for it_i, (source, target) in enumerate(
                batch_generator(
                    data,
                    sequence_length=sequence_length,
                    batch_size=batch_size)):
            mse_loss, mdn_loss, _ = sess.run(
                [net['mse_loss'], net['mdn_loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['keep_prob']: 0.8,
                    net['source']: source,
                    net['target']: target
                })
            total += mse_loss + mdn_loss
            total_mdn += mdn_loss
            total_mse += mse_loss
            print('{}: mdn: {}, mse: {}, total: {}'.format(
                it_i, mdn_loss, mse_loss, mdn_loss + mse_loss), end='\r')
        current_learning_rate = max(0.0001,
                                    current_learning_rate * 0.99)
        print('iteration: {}, learning rate: {}'.format(
            it_i, current_learning_rate))
        print('\n-- epoch {}: mdn: {}, mse: {}, total: {} --\n'.format(
            epoch_i, total_mdn / (it_i + 1), total_mse / (it_i + 1),
            total / (it_i + 1)))
        saver.save(sess, './seq2seq-mdn.ckpt', global_step=epoch_i)

    sess.close()


def euler():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    batch_size = 64
    sequence_length = 200
    n_features = data.shape[-1]
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 5
    use_attention = True
    use_mdn = True
    return locals()


def infer(data, mean_data, std_data, batch_size, sequence_length, **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default() as g, tf.Session(config=config) as sess:
        net = seq2seq.create_model(
            batch_size=batch_size,
            sequence_length=sequence_length,
            **kwargs)

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
            net['loss'])
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        source, target = next(
            batch_generator(
                data, sequence_length=sequence_length, batch_size=batch_size))
        recon = sess.run(
            net['decoding'],
            feed_dict={
                net['source']: source,
                net['keep_prob']: 1.0
            })[0]
        src = (source * std_data) + mean_data
        tgt = (target * std_data) + mean_data
        res = (recon * std_data) + mean_data
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(src[0])
        axs[0][0].set_title('Source')
        axs[0][1].plot(tgt[0])
        axs[0][1].set_title('Target (Original)')
        axs[1][0].plot(src[0])
        axs[1][0].set_title('Source')
        axs[1][1].plot(res[0])
        axs[1][1].set_title('Target (Synthesis)')


if __name__ == '__main__':
    params = euler()
    train(**params)
    infer(**params)
