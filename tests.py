import numpy as np
import train


def euler():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    n_features = data.shape[-1]
    batch_size = 50
    sequence_length = 240
    input_embed_size = 512
    n_neurons = 1024
    n_layers = 2
    n_gaussians = 5
    use_attention = True
    use_mdn = False
    # model_name = 'seq2seq.ckpt'
    restore_name = 'seq2seq.ckpt-508'

    train.infer(
        data=data,
        mean_data=mean_data,
        std_data=std_data,
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


def euler_v2():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    n_features = data.shape[-1]
    batch_size = 50
    sequence_length = 240
    input_embed_size = None
    n_neurons = 512
    n_layers = 2
    n_gaussians = 10
    use_attention = True
    use_mdn = False
    model_name = 'seq2seq-v2.ckpt-279'

    train.infer(
        data=data,
        mean_data=mean_data,
        std_data=std_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        input_embed_size=input_embed_size,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_attention=use_attention,
        use_mdn=use_mdn,
        model_name=model_name)


def euler_v3():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    n_features = data.shape[-1]
    sequence_length = 120
    input_embed_size = None
    n_neurons = 512
    n_layers = 3
    n_gaussians = 10
    use_attention = False
    use_mdn = False
    model_name = 'seq2seq-v3.ckpt-429'

    hop_length = 60
    idxs = np.arange(0, len(data) - sequence_length * 2, hop_length)
    source = np.array([data[i:i + sequence_length, :] for i in idxs])
    target = np.array([
        data[i + sequence_length:i + sequence_length * 2, :]
        for i in idxs
    ])
    batch_size = len(idxs)

    res = train.infer(
        source=source,
        target=target,
        mean_data=mean_data,
        std_data=std_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        input_embed_size=input_embed_size,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_attention=use_attention,
        use_mdn=use_mdn,
        model_name=model_name)


def euler_v4():
    data = np.load('euler.npy')
    data = data.reshape(data.shape[0], -1)
    mean_data = np.mean(data, axis=0)
    std_data = np.std(data, axis=0)
    data = (data - mean_data) / std_data
    n_features = data.shape[-1]
    batch_size = 20
    sequence_length = 500
    input_embed_size = None
    n_neurons = 512
    n_layers = 3
    n_gaussians = 10
    use_attention = False
    use_mdn = True
    model_name = 'seq2seq-v4'

    res = train.train(
        data=data,
        mean_data=mean_data,
        std_data=std_data,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_features=n_features,
        input_embed_size=input_embed_size,
        n_neurons=n_neurons,
        n_layers=n_layers,
        n_gaussians=n_gaussians,
        use_attention=use_attention,
        use_mdn=use_mdn,
        model_name=model_name)


if __name__ == '__main__':
    euler_v4()
