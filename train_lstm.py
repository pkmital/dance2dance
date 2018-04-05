import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import lstm_mdn


def batch_generator(data, sequence_length, batch_size=50):
    idxs = np.random.permutation(np.arange(len(data) - sequence_length))
    n_batches = max(1, len(idxs) // (batch_size * sequence_length))
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
          overfit=False,
          restore_name=None,
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
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
            it_i = 0
            total_loss, total_mse, total_weighted_mse = 0.0, 0.0, 0.0
            if overfit:
                source = data[:sequence_length, :].reshape(
                    1, sequence_length, -1)
                source = np.tile(source, [batch_size, 1, 1])
                weighted_mse, mse, loss, _ = sess.run(
                    [net['weighted_mse'], net['mse'], net['loss'], opt],
                    feed_dict={
                        learning_rate: current_learning_rate,
                        net['keep_prob']: 0.8,
                        net['source']: source
                    })
                total_loss += loss
                total_mse += mse
                total_weighted_mse += weighted_mse
                print(
                    'loss: {} mse: {} weighted_mse: {}'.format(
                        loss, mse, weighted_mse),
                    end='\r')
            else:
                for it_i, source in enumerate(
                        batch_generator(
                            data,
                            sequence_length=sequence_length,
                            batch_size=batch_size)):
                    weighted_mse, loss, _ = sess.run(
                        [net['weighted_mse'], net['loss'], opt],
                        feed_dict={
                            learning_rate: current_learning_rate,
                            net['keep_prob']: 0.8,
                            net['source']: source
                        })
                    total_loss += loss
                    total_weighted_mse += weighted_mse
                    print(
                        '{}: total_loss: {} total_weighted_mse: {}'.
                        format(it_i, loss, weighted_mse),
                        end='\r')
            current_learning_rate = max(0.0001, current_learning_rate * 0.995)
            print('iteration: {}, learning rate: {}'.format(
                it_i, current_learning_rate))
            print(
                '\n-- epoch {}: total_loss: {} total_mse: {} total_weighted_mse: {} --\n'.
                format(epoch_i, total_loss / (it_i + 1), total_mse / (it_i + 1),
                       total_weighted_mse / (it_i + 1)))
            saver.save(
                sess, os.path.join(ckpt_path, model_name), global_step=epoch_i)


def infer(source,
          data_mean,
          data_std,
          batch_size,
          sequence_length,
          n_features,
          n_neurons,
          n_layers,
          n_gaussians,
          use_mdn,
          prime_length,
          ckpt_path='./',
          restore_name='lstm_mdn.ckpt',
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        net = lstm_mdn.create_model(
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_neurons=n_neurons,
            n_features=n_features,
            n_layers=n_layers,
            n_gaussians=n_gaussians,
            use_mdn=use_mdn)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_path, restore_name))
        # Prime
        sample, initial_state = sess.run(
            [net['sample'], net['final_state']],
            feed_dict={
                net['source']: source[:, :prime_length, :],
                net['keep_prob']: 1.0
            })
        decoding = sample[:, [-1], :]

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        net = lstm_mdn.create_model(
            batch_size=batch_size,
            sequence_length=1,
            n_neurons=n_neurons,
            n_features=n_features,
            n_layers=n_layers,
            n_gaussians=n_gaussians,
            use_mdn=use_mdn)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(ckpt_path, restore_name))
        prediction_weighted, prediction_sampled = [], []
        # Generate
        for i in range(prime_length, prime_length + sequence_length):
            print('{} / {}'.format(i, sequence_length + prime_length), end='\r')
            initial_state = np.array(initial_state).reshape(-1, 2, 1, n_neurons)
            weighted, sample, initial_state = sess.run(
                [
                    net['weighted_reconstruction'], net['sample'],
                    net['final_state']
                ],
                feed_dict={
                    net['source']: decoding,
                    net['keep_prob']: 1.0,
                    net['initial_state']: initial_state
                })
            prediction_weighted.append(weighted)
            prediction_sampled.append(sample)
        src = (source[:, :prime_length, :] * data_std) + data_mean
        tgt = (source[:, prime_length:, :] * data_std) + data_mean
        res_sample = (np.squeeze(prediction_sampled) * data_std) + data_mean
        res_weight = (np.squeeze(prediction_weighted) * data_std) + data_mean
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(src.reshape(-1, n_features))
        axs[0][0].set_title('Prime')
        axs[0][1].plot(tgt.reshape(-1, n_features))
        axs[0][1].set_title('Target (Original)')
        axs[1][0].plot(res_weight.reshape(-1, n_features))
        axs[1][0].set_title('Target (No Sampling)')
        axs[1][1].plot(res_sample.reshape(-1, n_features))
        axs[1][1].set_title('Target (Sampling)')
        np.save('source.npy', src)
        np.save('target.npy', tgt)
        np.save('prediction_weighted.npy', res_weight)
        np.save('prediction_sampled.npy', res_sample)


def test_euler():
    data = np.load('euler.npy')
    data = data.reshape(data.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    idxs = np.where(data_std > 0)[0]
    data_mean = data_mean[idxs]
    data_std = data_std[idxs]
    data = (data[:, idxs] - data_mean) / data_std
    batch_size = 64
    sequence_length = 256
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 10
    use_mdn = True
    model_name = 'lstm_mdn-euler'
    restore_name = 'lstm_mdn-euler-365'
    overfit = False

    if overfit:
        data = data[:, [0]]
        data_mean = data_mean[[0]]
        data_std = data_std[[0]]
    n_features = data.shape[-1]

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
        overfit=overfit,
        model_name=model_name,
        restore_name=restore_name)

    batch_size = 1
    sequence_length = 120
    source = data[:sequence_length * 2, :].reshape(1, sequence_length * 2, -1)
    res = infer(
        source=source,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prime_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        restore_name=restore_name)
    return res


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
    restore_name = 'lstm_mdn-quats-16'

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
        model_name=model_name,
        restore_name=restore_name)

    batch_size = 1
    source = data[:sequence_length * 2, :].reshape(1, sequence_length * 2, -1)
    res = infer(
        source=source,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prime_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        restore_name=restore_name)
    return res


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
    restore_name = 'lstm_mdn-local-positions-60'

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
        model_name=model_name,
        restore_name=restore_name)

    batch_size = 1
    source = data[:sequence_length * 2, :].reshape(1, sequence_length * 2, -1)
    res = infer(
        source=source,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        prime_length=sequence_length,
        n_features=n_features,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_mdn=use_mdn,
        restore_name=restore_name)
    return res
