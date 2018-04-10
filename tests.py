import numpy as np
import train


def test_euler():
    data = np.load('euler.npy')
    data = data.reshape(data.shape[0], -1)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    idxs = np.where(data_std > 0)[0]
    data_mean = data_mean[idxs]
    data_std = data_std[idxs]
    data = (data[:, idxs] - data_mean) / data_std
    n_features = data.shape[-1]
    batch_size = 100
    sequence_length = 120
    input_embed_size = None
    n_neurons = 1024
    n_layers = 3
    n_gaussians = 24
    use_attention = True
    use_mdn = True
    n_epochs = 5000
    model_name = 'seq2seq-24-gaussians-3x1024'
    restore_name = None

    res = train.train(
        data=data,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        input_embed_size=input_embed_size,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_attention=use_attention,
        use_mdn=use_mdn,
        n_epochs=n_epochs,
        model_name=model_name,
        restore_name=restore_name)

    batch_size = 1
    offset = 0
    source = data[offset:offset + sequence_length * batch_size, :].reshape(
        batch_size, sequence_length, -1)
    target = data[offset + sequence_length * batch_size:offset + sequence_length * batch_size * 2, :].reshape(
        batch_size, sequence_length, -1)
    restore_name = 'seq2seq-20-gaussians-3x1024-571'

    train.infer(
        source=source,
        target=target,
        data_mean=data_mean,
        data_std=data_std,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        input_embed_size=input_embed_size,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_attention=use_attention,
        use_mdn=use_mdn,
        model_name=restore_name)


if __name__ == '__main__':
    test_euler()
