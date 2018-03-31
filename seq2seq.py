import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.distributions as tfd


def gausspdf(x, mean, sigma):
    return tf.exp(-(x - mean)**2 /
                  (2 * sigma**2)) / (tf.sqrt(2.0 * np.pi) * sigma)


def _make_pos_def(mat):
    mat = (mat + tf.transpose(mat, perm=[0, 1, 3, 2])) / 2.
    e, v = tf.self_adjoint_eig(mat)
    e = tf.where(e > 1e-14, e, 1e-14 * tf.ones_like(e))
    mat_pos_def = tf.matmul(
        tf.matmul(v, tf.matrix_diag(e), transpose_a=True), v)
    return mat_pos_def


def _create_embedding(x, embed_size, embed_matrix=None):
    batch_size, sequence_length, n_input = x.shape.as_list()
    # Creating an embedding matrix if one isn't given
    if embed_matrix is None:
        embed_matrix = tf.get_variable(
            name='embed_matrix',
            shape=[n_input, embed_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
    embed = tf.reshape(
        tf.matmul(
            tf.reshape(x, [batch_size * sequence_length, n_input]),
            embed_matrix), [batch_size, sequence_length, embed_size])
    return embed, embed_matrix


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
    (outputs_fw, output_bw), (final_state_fw, final_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=source,
        sequence_length=lengths,
        time_major=False,
        dtype=tf.float32)

    return outputs_fw, final_state_fw


def _create_decoder(n_dec_neurons,
                    n_layers,
                    keep_prob,
                    batch_size,
                    encoder_outputs,
                    encoder_state,
                    encoder_lengths,
                    decoding_inputs,
                    decoding_lengths,
                    n_features,
                    scope,
                    max_sequence_size,
                    use_attention=False):
    from tensorflow.python.layers.core import Dense
    output_layer = Dense(n_features, name='output_projection')

    with tf.variable_scope('forward'):
        cells = _create_rnn_cell(n_dec_neurons, n_layers, keep_prob)

    if use_attention:
        attn_mech = tf.contrib.seq2seq.LuongAttention(
            cells.output_size, encoder_outputs, encoder_lengths, scale=False)
        cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=cells,
            attention_mechanism=attn_mech,
            attention_layer_size=cells.output_size,
            alignment_history=False)
        initial_state = cells.zero_state(
            dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)
    else:
        initial_state = encoder_state

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=decoding_inputs,
        sequence_length=decoding_lengths,
        time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)

    return outputs


def create_model(batch_size=50,
                 sequence_length=120,
                 n_features=72,
                 input_embed_size=512,
                 target_embed_size=512,
                 share_input_and_target_embedding=True,
                 n_neurons=512,
                 n_layers=2,
                 n_gaussians=5,
                 use_attention=False):
    # [batch_size, max_time, n_features]
    source = tf.placeholder(
        tf.float32,
        shape=(batch_size, sequence_length, n_features),
        name='source')
    target = tf.placeholder(
        tf.float32,
        shape=(batch_size, sequence_length, n_features),
        name='source')
    lengths = tf.multiply(
        tf.ones((batch_size,), tf.int32),
        sequence_length - 1,
        name='source_lengths')

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Get the input to the decoder by removing last element
    # and adding a 'go' symbol as first element
    with tf.variable_scope('target/slicing'):
        decoder_input = tf.slice(target, [0, 0, 0],
                                 [batch_size, sequence_length - 1, n_features])
        decoder_output = tf.slice(target, [0, 1, 0],
                                  [batch_size, sequence_length - 1, n_features])

    # Embed word ids to target embedding
    with tf.variable_scope('source/embedding'):
        source_embed, source_embed_matrix = _create_embedding(
            x=source, embed_size=input_embed_size)

    # Embed word ids for target embedding
    with tf.variable_scope('target/embedding'):
        # Check if we need a new embedding matrix or not.  If we are for
        # instance translating to another language, then we'd need different
        # vocabularies for the input and outputs, and so new embeddings.
        # However if we are for instance building a chatbot with the same
        # language, then it doesn't make sense to have different embeddings and
        # we should share them.
        if share_input_and_target_embedding:
            target_input_embed, target_embed_matrix = _create_embedding(
                x=decoder_input,
                embed_size=target_embed_size,
                embed_matrix=source_embed_matrix)
        else:
            target_input_embed, target_embed_matrix = _create_embedding(
                x=decoder_input, embed_size=target_embed_size)

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_state = _create_encoder(
            source=source_embed,
            lengths=lengths,
            batch_size=batch_size,
            n_enc_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob)

    # TODO: Add (vq?) variational loss

    # Build the decoder
    with tf.variable_scope('decoder') as scope:
        n_outputs = n_features * n_gaussians + (
            n_features * n_features) * n_gaussians + n_gaussians
        decoding = _create_decoder(
            n_dec_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob,
            batch_size=batch_size,
            encoder_outputs=encoder_outputs,
            encoder_state=encoder_state,
            encoder_lengths=lengths,
            decoding_inputs=decoder_input,
            decoding_lengths=lengths,
            n_features=n_outputs,
            scope=scope,
            max_sequence_size=sequence_length - 1)

    with tf.variable_scope('mdn'):
        means = tf.reshape(
            tf.slice(
                decoding[0], [0, 0, 0],
                [batch_size, sequence_length - 1, n_features * n_gaussians]),
            [batch_size, sequence_length - 1, n_features, n_gaussians])
        sigmas = tf.reshape(
            tf.slice(decoding[0], [0, 0, n_features * n_gaussians], [
                batch_size, sequence_length - 1,
                n_features * n_features * n_gaussians
            ]), [
                batch_size, sequence_length - 1, n_features, n_features,
                n_gaussians
            ])
        weights = tf.reshape(
            tf.slice(
                decoding[0],
                [0, 0, n_features * n_gaussians + n_features * n_features * n_gaussians],
                [batch_size, sequence_length - 1, n_gaussians]),
            [batch_size, sequence_length - 1, n_gaussians])
        components = []
        for gauss_i in range(n_gaussians):
            mean_i = means[:, :, :, gauss_i]
            sigma_i = sigmas[:, :, :, :, gauss_i]
            sigma_i = _make_pos_def(sigma_i)
            components.append(
                tfd.MultivariateNormalFullCovariance(
                    loc=mean_i, covariance_matrix=sigma_i))
        gauss = tfd.Mixture(
            cat=tfd.Categorical(probs=weights), components=components)

    with tf.variable_scope('loss'):
        p = gauss.prob(decoder_output)
        negloglike = -tf.log(tf.maximum(p, 1e-10))
        loss = tf.reduce_mean(tf.reduce_sum(negloglike, 1))

    return {
        'source': source,
        'target': target,
        'keep_prob': keep_prob,
        'encoding': encoder_state,
        'decoding': decoding,
        'p': p,
        'loss': loss
    }
