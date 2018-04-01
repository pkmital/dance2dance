from functools import partial
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.distributions as tfd


class RegressionHelper(tf.contrib.seq2seq.Helper):
    """Helper interface.    Helper instances are used by SamplingDecoder."""
    def __init__(self, batch_size, max_sequence_size, n_features):
        self._batch_size = batch_size
        self._max_sequence_size = max_sequence_size
        self._n_features = n_features
        self._batch_size_tensor = tf.convert_to_tensor(
                batch_size, dtype=tf.int32, name="batch_size")

    @property
    def batch_size(self):
        """Returns a scalar int32 tensor."""
        return self._batch_size_tensor

    @property
    def sample_ids_dtype(self):
        return tf.float32

    @property
    def sample_ids_shape(self):
        return self._n_features

    def initialize(self, name=None):
        finished = tf.tile([False], [self._batch_size])
        start_inputs = tf.fill([self._batch_size, self._n_features], 0.0)
        return (finished, start_inputs)

    def sample(self, time, outputs, state, name=None):
        """Returns `sample_ids`."""
        del time, state
        return outputs

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """Returns `(finished, next_inputs, next_state)`."""
        del sample_ids
        finished = tf.cond(
            tf.less(time, self._max_sequence_size),
            lambda: False, lambda: True)
        del time
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: tf.zeros_like(outputs),
            lambda: outputs)
        return (finished, next_inputs, state)


def gausspdf(x, mean, sigma):
    return -(x - mean)**2 / (2 * sigma**2)


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
    (outputs_fw, output_bw), (final_state_fw, final_state_bw) = \
        tf.nn.bidirectional_dynamic_rnn(
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
                    n_gaussians,
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

    helper = RegressionHelper(
        batch_size=batch_size, max_sequence_size=max_sequence_size, n_features=n_features)
    scope.reuse_variables()
    infer_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell=cells,
        helper=helper,
        initial_state=initial_state,
        output_layer=output_layer)
    infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        infer_decoder,
        output_time_major=False,
        impute_finished=True,
        maximum_iterations=max_sequence_size)
    # infer_logits = tf.identity(infer_outputs.sample_id, name='infer_logits')
    return outputs, infer_outputs


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
        name='target')
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
        outputs, infer_outputs = _create_decoder(
            n_dec_neurons=n_neurons,
            n_layers=n_layers,
            keep_prob=keep_prob,
            batch_size=batch_size,
            encoder_outputs=encoder_outputs,
            encoder_state=encoder_state,
            encoder_lengths=lengths,
            decoding_inputs=decoder_input,
            decoding_lengths=lengths,
            n_features=n_features,
            scope=scope,
            max_sequence_size=sequence_length - 1,
            n_gaussians=n_gaussians)

    with tf.variable_scope('loss'):
        mdn_loss = tf.reduce_mean(tf.reduce_sum([[0.0]], 1))
        mse_loss = tf.losses.mean_squared_error(
            outputs[0], decoder_output)
        loss = mdn_loss + mse_loss

    return {
        'source': source,
        'target': target,
        'keep_prob': keep_prob,
        'encoding': encoder_state,
        'decoding': infer_outputs,
        'loss': loss,
        'mdn_loss': mdn_loss,
        'mse_loss': mse_loss
    }
