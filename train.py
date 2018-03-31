import numpy as np
import tensorflow as tf
import seq2seq


def batch_generator(data, sequence_length, batch_size=50):
    idxs = np.random.permutation(np.arange(len(data) - sequence_length * 2))
    n_batches = len(idxs) // (batch_size * sequence_length)
    for batch_i in range(n_batches):
        this_idxs = idxs[batch_i * batch_size:(batch_i + 1) * batch_size]
        source = [data[i:i + sequence_length, :] for i in this_idxs]
        target = [data[i + sequence_length:i + sequence_length * 2, :] for i in this_idxs]
        yield np.array(source, dtype=np.float32), np.array(target, dtype=np.float32)


def train(data, n_epochs=1000, batch_size=100, sequence_length=240, **kwargs):
    n_features = data.shape[1]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = seq2seq.create_model(
        batch_size=batch_size,
        n_features=n_features,
        sequence_length=sequence_length)

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        net['loss'])
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    saver = tf.train.Saver()

    current_learning_rate = 0.0001
    for epoch_i in range(n_epochs):
        total, total_mse, total_mdn = 0, 0, 0
        for it_i, (source, target) in enumerate(
                batch_generator(
                    data,
                    sequence_length=sequence_length,
                    batch_size=batch_size)):
            if it_i % 1000 == 0:
                current_learning_rate = max(0.0001,
                                            current_learning_rate * 0.99)
                print(it_i)
            mse_loss, mdn_loss, _ = sess.run(
                [net['mse_loss'], net['mdn_loss'], opt],
                feed_dict={
                    learning_rate: current_learning_rate,
                    net['keep_prob']: 0.8,
                    net['source']: source,
                    net['target']: target
                })
            total = total + mse_loss + mdn_loss
            total_mdn = total_mdn + mdn_loss
            total_mse = total_mse + mse_loss
            print('{}: mdn: {}, mse: {}, total: {}'.format(
                it_i,
                total_mdn / (it_i + 1),
                total_mse / (it_i + 1),
                total / (it_i + 1)), end='\r')
        # End of epoch, save
        print('epoch {}: {}'.format(epoch_i, total / it_i))
        saver.save(sess, './seq2seq.ckpt', global_step=it_i)

    sess.close()


def run_euler():
    data = np.load('quats.npy')
    data = (data.reshape([data.shape[0], -1]) - np.mean(data)) / np.std(data)
    batch_size = 32
    sequence_length = 120
    n_features = data.shape[-1]
    input_embed_size = 512
    target_embed_size = 512
    share_input_and_target_embedding = True
    n_neurons = 512
    n_layers = 2
    n_gaussians = 10
    use_attention = True
    train(**locals())


if __name__ == '__main__':
    run_euler()
