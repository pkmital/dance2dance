import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import lstm_mdn

def batch_generator(data, sequence_length, batch_size=50):
    idxs = np.random.permutation(np.arange(len(data) - sequence_length))
    n_batches = len(idxs) // (batch_size * sequence_length)
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        source = [data[i:i + sequence_length, :] for i in this_idxs]
        yield np.array(source, dtype=np.float32)


def train(data,
          data_mean,
          data_std,
          n_epochs=1000,
          batch_size=100,
          sequence_length=240,
          ckpt_path='./',
          model_name='lstm_mdn.ckpt',
          restore_name=None,
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = lstm_mdn.create_model(
        batch_size=batch_size, sequence_length=sequence_length, **kwargs)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()
    if restore_name:
        saver.restore(sess, restore_name)

    current_learning_rate = 0.01
    for epoch_i in range(n_epochs):
        total = 0.0
        for it_i, source in enumerate(
                batch_generator(
                    data,
                    sequence_length=sequence_length,
                    batch_size=batch_size)):
            loss, _ = sess.run(
                [net['loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['keep_prob']: 0.8,
                    net['source']: source
                })
            total += loss
            print('{}: total: {}'.format(it_i, loss), end='\r')
        current_learning_rate = max(0.0001, current_learning_rate * 0.99)
        print('iteration: {}, learning rate: {}'.format(it_i,
                                                        current_learning_rate))
        print('\n-- epoch {}: total: {} --\n'.format(
            epoch_i, total / (it_i + 1)))
        saver.save(
            sess, os.path.join(ckpt_path, model_name), global_step=epoch_i)

    sess.close()


def infer(source,
          target,
          data_mean,
          data_std,
          batch_size,
          sequence_length,
          ckpt_path='./',
          model_name='lstm_mdn.ckpt',
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default() as g, tf.Session(config=config) as sess:
        net = lstm_mdn.create_model(
            batch_size=batch_size, sequence_length=sequence_length, **kwargs)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_path, model_name))
        recon, enc = sess.run(
            [net['decoding'], net['encoding']],
            feed_dict={
                net['source']: source,
                net['keep_prob']: 1.0
            })
        src = (source * data_std) + data_mean
        tgt = (target * data_std) + data_mean
        res = (recon[0] * data_std) + data_mean
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(src.reshape(-1, src.shape[-1]))
        axs[0][0].set_title('Source')
        axs[0][1].plot(tgt.reshape(-1, tgt.shape[-1]))
        axs[0][1].set_title('Target (Original)')
        axs[1][0].plot(src.reshape(-1, src.shape[-1]))
        axs[1][0].set_title('Source')
        axs[1][1].plot(res.reshape(-1, res.shape[-1]))
        axs[1][1].set_title('Target (Synthesis)')
        np.save('source.npy', src)
        np.save('target.npy', tgt)
        np.save('encoding.npy', enc)
        np.save('prediction.npy', res)
        return {
            'source': src,
            'target': tgt,
            'encoding': enc,
            'prediction': res
        }


def test_euler():
    data = np.load('euler.npy')
    data = data.reshape(data.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    idxs = np.where(data_std > 0)[0]
    data = (data[:, idxs] - data_mean[idxs]) / data_std[idxs]
    n_features = data.shape[-1]
    batch_size = 20
    sequence_length = 240
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 10
    use_mdn = True
    model_name = 'lstm_mdn-euler' # v1

    res = train(
        data=data,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        model_name=model_name)


def test_quats():
    data = np.load('quats.npy')
    data = data.reshape(data.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    idxs = np.where(data_std > 0)[0]
    data = (data[:, idxs] - data_mean[idxs]) / data_std[idxs]
    n_features = data.shape[-1]
    batch_size = 20
    sequence_length = 240
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 10
    use_mdn = True
    model_name = 'lstm_mdn-quats'

    res = train(
        data=data,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        model_name=model_name)


def test_local_positions():
    data = np.load('local_positions.npy')
    data = data.reshape(data.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    idxs = np.where(data_std > 0)[0]
    data = (data[:, idxs] - data_mean[idxs]) / data_std[idxs]
    n_features = data.shape[-1]
    batch_size = 20
    sequence_length = 240
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 10
    use_mdn = True
    model_name = 'lstm_mdn-local-positions'

    res = train(
        data=data,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        model_name=model_name)

