import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def flatten(x, name=None, reuse=None):
    """Flatten Tensor to 2-dimensions.
    Parameters
    ----------
    x : tf.Tensor
        Input tensor to flatten.
    name : None, optional
        Variable scope for flatten operations
    Returns
    -------
    flattened : tf.Tensor
        Flattened tensor.
    """
    with tf.variable_scope('flatten'):
        dims = x.get_shape().as_list()
        if len(dims) >= 3:
            flattened = tf.reshape(x, shape=[-1, np.prod(dims[1:])])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2, 3 or 4.  Found:',
                             len(dims))
        return flattened


def _create_embedding(x, embed_size, embed_matrix=None):
    if len(x.get_shape()) != 2:
        x = flatten(x, reuse=reuse)

    n_input = x.get_shape().as_list()[1]

    # Creating an embedding matrix if one isn't given
    if embed_matrix is None:
        embed_matrix = tf.get_variable(
            name='embed_matrix',
            shape=[n_input, embed_size],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

    embed = tf.matmul(x, embed_matrix)

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
    (outputs, final_state) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=source,
        sequence_length=lengths,
        time_major=False,
        dtype=tf.float32)

    return outputs, final_state


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
                    use_attention=True):
    with tf.variable_scope('forward'):
        cell_fw = _create_rnn_cell(n_enc_neurons, n_layers, keep_prob)

    if use_attention:
        attn_mech = tf.contrib.seq2seq.LuongAttention(
            cell_fw.output_size, encoder_outputs, encoder_lengths, scale=True)
        cells = tf.contrib.seq2seq.AttentionWrapper(
            cell=cell_fw,
            attention_mechanism=attn_mech,
            attention_layer_size=cell_fw.output_size,
            alignment_history=False)
        initial_state = cells.zero_state(dtype=tf.float32, batch_size=batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)
    else:
        initial_state = None

    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs=decoding_inputs,
        sequence_length=decoding_lengths,
        time_major=False)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells_fw,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)

    return outputs


def create_model(input_embed_size=512,
                 target_embed_size=512,
                 share_input_and_target_embedding=True,
                 n_neurons=512,
                 n_layers=4,
                 n_gaussians=10,
                 use_attention=True,
                 max_sequence_size=30):
    n_enc_neurons = n_neurons
    n_dec_neurons = n_neurons

    # [batch_size, max_time, n_features]
    source = tf.placeholder(tf.int32, shape=(None, None, None), name='source')
    source_lengths = tf.placeholder(
        tf.int32, shape=(None,), name='source_lengths')

    # [batch_size, max_time, n_features]
    target = tf.placeholder(tf.int32, shape=(None, None, None), name='target')
    target_lengths = tf.placeholder(
        tf.int32, shape=(None,), name='target_lengths')

    # Dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Symbolic shapes
    batch_size, sequence_length, n_features = tf.unstack(tf.shape(source))

    # Get the input to the decoder by removing last element
    # and adding a 'go' symbol as first element
    with tf.variable_scope('target/slicing'):
        slice = tf.slice(target, [0, 0], [batch_size, -1])
        decoder_input = tf.concat([tf.fill([batch_size, 1], GO_ID), slice], 1)

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
                x=target, embed_size=target_embed_size)

    # Build the encoder
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_state = _create_encoder(
            source=source_embed,
            lengths=source_lengths,
            batch_size=batch_size,
            n_enc_neurons=n_enc_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob)

    # TODO: Add (vq?) variational loss

    # Build the decoder
    with tf.variable_scope('decoder') as scope:
        decoding = _create_decoder(
            n_dec_neurons=n_dec_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob
            batch_size=batch_size,
            encoder_outputs=encoder_outputs[0],
            encoder_state=encoder_state[0],
            encoder_lengths=source_lengths,
            decoding_inputs=decoder_input,
            decoding_lengths=target_lengths,
            n_features=n_features,
            scope=scope,
            max_sequence_size=max_sequence_size)

    with tf.variable_scope('mdn'):
        means = tf.reshape(
            tfl.linear(
                inputs=decoding,
                num_outputs=n_output_features * n_gaussians,
                activation_fn=None,
                scope='means'), [-1, n_features, n_gaussians])
        sigmas = tf.maximum(
            tf.reshape(
                tfl.linear(
                    inputs=current_input,
                    num_outputs=n_features * n_gaussians,
                    activation_fn=tf.nn.relu,
                    scope='sigmas'), [-1, n_features, n_gaussians]), 1e-10)
        weights = tf.reshape(
            tfl.linear(
                inputs=current_input,
                num_outputs=n_features * n_gaussians,
                activation_fn=tf.nn.softmax,
                scope='weights'), [-1, n_features, n_gaussians])

    with tf.variable_scope('loss'):
        p = gausspdf(decoder_output, means, sigmas)
        weighted = weights * p
        sump = tf.reduce_sum(weighted, axis=2)
        negloglike = -tf.log(tf.maximum(sump, 1e-10))
        cost = tf.reduce_mean(tf.reduce_mean(negloglike, 1))

    return {
        'loss': loss,
        'source': source,
        'source_lengths': source_lengths,
        'target': target,
        'target_lengths': target_lengths,
        'keep_prob': keep_prob,
        'thought_vector': encoder_state,
        'decoder': decoding_infer_logits
    }
