import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.layers as tfl


def gausspdf(x, mean, sigma):
    return tf.exp(-(x - mean)**2 /
                  (2 * sigma**2)) / (tf.sqrt(2.0 * np.pi) * sigma)


def _create_rnn_cell(n_neurons, n_layers, keep_prob):
    cell_fw = rnn.LayerNormBasicLSTMCell(
        num_units=n_neurons, dropout_keep_prob=keep_prob)
    # Build deeper recurrent net if using more than 1 layer
    if n_layers > 1:
        cells = [cell_fw]
        for layer_i in range(1, n_layers):
            with tf.variable_scope('{}'.format(layer_i)):
                cell_fw = rnn.LayerNormBasicLSTMCell(
                    num_units=n_neurons, dropout_keep_prob=keep_prob)
                cells.append(cell_fw)
        cell_fw = rnn.MultiRNNCell(cells)
    return cell_fw


def _create_encoder(source, lengths, batch_size, n_enc_neurons, n_layers,
                    keep_prob):
    # Create the RNN Cells for encoder
    with tf.variable_scope('forward'):
        cell_fw = _create_rnn_cell(n_enc_neurons, n_layers, keep_prob)

    # Create the internal multi-layer cell for the backward RNN.
    with tf.variable_scope('backward'):
        cell_bw = _create_rnn_cell(n_enc_neurons, n_layers, keep_prob)

    # Now hookup the cells to the input
    # [batch_size, max_time, embed_size]
    (outputs_fw, output_bw), (final_state_fw, final_state_bw) = \
        tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=source,
            sequence_length=lengths,
            time_major=False,
            dtype=tf.float32)

    return outputs_fw, final_state_fw


def create_model(batch_size=50,
                 sequence_length=120,
                 n_features=72,
                 n_neurons=512,
                 n_layers=2,
                 n_gaussians=5,
                 use_mdn=False):
    # [batch_size, max_time, n_features]
    source = tf.placeholder(
        tf.float32,
        shape=(batch_size, sequence_length, n_features),
        name='source')
    lengths = tf.multiply(
        tf.ones((batch_size,), tf.int32),
        sequence_length,
        name='source_lengths')

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    with tf.variable_scope('target/slicing'):
        source_input = tf.slice(source, [0, 0, 0],
                                 [batch_size, sequence_length - 1, n_features])
        source_output = tf.slice(source, [0, 1, 0],
                                  [batch_size, sequence_length - 1, n_features])

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoding, state = _create_encoder(
            source=source_input,
            lengths=lengths,
            batch_size=batch_size,
            n_enc_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob)

    n_outputs = n_features * n_gaussians + n_features * n_gaussians + n_gaussians
    outputs = tfl.fully_connected(encoding, n_outputs, activation_fn=None)

    # TODO: Add (vq?) variational loss

    max_sequence_size = sequence_length - 1
    with tf.variable_scope('mdn'):
        means = tf.reshape(
            tf.slice(
                outputs, [0, 0, 0],
                [batch_size, max_sequence_size, n_features * n_gaussians]),
            [batch_size, max_sequence_size, n_features, n_gaussians])
        sigmas = tf.nn.softplus(
            tf.reshape(
                tf.slice(outputs, [0, 0, n_features * n_gaussians], [
                    batch_size, max_sequence_size, n_features * n_gaussians
                ]),
                [batch_size, max_sequence_size, n_features, n_gaussians]))
        weights = tf.nn.softmax(
            tf.reshape(
                tf.slice(outputs, [
                    0, 0,
                    n_features * n_gaussians + n_features * n_gaussians
                ], [batch_size, max_sequence_size, n_gaussians]),
                [batch_size, max_sequence_size, n_gaussians]))
        components = []
        for gauss_i in range(n_gaussians):
            mean_i = means[:, :, :, gauss_i]
            sigma_i = sigmas[:, :, :, gauss_i]
            components.append(
                tfd.MultivariateNormalDiag(
                    loc=mean_i, scale_diag=sigma_i))
        gauss = tfd.Mixture(
            cat=tfd.Categorical(probs=weights), components=components)

    with tf.variable_scope('loss'):
        p = gauss.log_prob(source_output)
        negloglike = -tf.reduce_logsumexp(p, axis=1)
        mdn_loss = tf.reduce_mean(negloglike)
        loss = mdn_loss

    return {
        'source': source,
        'keep_prob': keep_prob,
        'decoding': outputs,
        'loss': loss
    }
