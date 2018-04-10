import os
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


def fixed_generator(data, hop_length, sequence_length, batch_size=50):
    idxs = np.arange(0, len(data) - sequence_length, hop_length)
    n_batches = len(idxs)
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


def train(data,
          data_mean,
          data_std,
          n_epochs=1000,
          batch_size=100,
          sequence_length=240,
          ckpt_path='./',
          model_name='seq2seq.ckpt',
          restore_name=None,
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    net = seq2seq.create_model(
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

    current_learning_rate = 0.0001
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
            print(
                '{}: mdn: {}, mse: {}, total: {}'.format(
                    it_i, mdn_loss, mse_loss, mdn_loss + mse_loss),
                end='\r')
        current_learning_rate = max(0.0001, current_learning_rate * 0.99)
        print('iteration: {}, learning rate: {}'.format(it_i,
                                                        current_learning_rate))
        print('\n-- epoch {}: mdn: {}, mse: {}, total: {} --\n'.format(
            epoch_i, total_mdn / (it_i + 1), total_mse / (it_i + 1),
            total / (it_i + 1)))
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
          model_name='seq2seq.ckpt',
          **kwargs):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default() as g, tf.Session(config=config) as sess:
        net = seq2seq.create_model(
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
        res = np.minimum(1.0, np.maximum(0.0, (recon[1] * data_std) + data_mean))
        fig, axs = plt.subplots(1, 3, sharey=True)
        axs[0].plot(src.reshape(-1, src.shape[-1]))
        axs[0].set_title('Source')
        axs[1].plot(tgt.reshape(-1, tgt.shape[-1]))
        axs[1].set_title('Target (Original)')
        axs[2].plot(res.reshape(-1, res.shape[-1]))
        axs[2].set_title('Target (Synthesis Sampling)')
        return {
            'source': src,
            'target': tgt,
            'encoding': enc,
            'prediction': res
        }


# def feedback(source,
#              target,
#              data_mean,
#              data_std,
#              batch_size,
#              sequence_length,
#              ckpt_path='./',
#              model_name='seq2seq.ckpt',
#              **kwargs):
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     with tf.Graph().as_default() as g, tf.Session(config=config) as sess:
#         net = seq2seq.create_model(
#             batch_size=batch_size, sequence_length=sequence_length, **kwargs)
# 
#         init_op = tf.group(tf.global_variables_initializer(),
#                            tf.local_variables_initializer())
#         sess.run(init_op)
#         saver = tf.train.Saver()
#         saver.restore(sess, os.path.join(ckpt_path, model_name))
#         recon, enc = sess.run(
#             [net['decoding'], net['encoding']],
#             feed_dict={
#                 net['source']: source,
#                 net['keep_prob']: 1.0
#             })
#         src = (source * data_std) + data_mean
#         tgt = (target * data_std) + data_mean
#         res = np.minimum(1.0, np.maximum(0.0, (recon[1] * data_std) + data_mean))
#         fig, axs = plt.subplots(1, 3, sharey=True)
#         axs[0].plot(src.reshape(-1, src.shape[-1]))
#         axs[0].set_title('Source')
#         axs[1].plot(tgt.reshape(-1, tgt.shape[-1]))
#         axs[1].set_title('Target (Original)')
#         axs[2].plot(res.reshape(-1, res.shape[-1]))
#         axs[2].set_title('Target (Synthesis Sampling)')
#         np.save('source.npy', src)
#         np.save('target.npy', tgt)
#         np.save('encoding.npy', enc)
#         np.save('prediction.npy', res)
#         return {
#             'source': src,
#             'target': tgt,
#             'encoding': enc,
#             'prediction': res
#         }
