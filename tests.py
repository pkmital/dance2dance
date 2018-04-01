import numpy as np
import train


def euler():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    n_features = data.shape[-1]
    batch_size = 64
    sequence_length = 200
    input_embed_size = 512
    n_neurons = 1024
    n_layers = 2
    n_gaussians = 5
    use_attention = True
    use_mdn = False
    model_name = 'seq2seq.ckpt'
    return locals()


def euler_v2():
    data = np.load('euler.npy')
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data.reshape([data.shape[0], -1]) - mean_data) / std_data
    n_features = data.shape[-1]
    batch_size = 32
    sequence_length = 120
    input_embed_size = None
    n_neurons = 512
    n_layers = 2
    n_gaussians = 10
    use_attention = True
    use_mdn = False
    model_name = 'seq2seq-v2.ckpt'
    return locals()


if __name__ == '__main__':
    params = euler_v2()
    train.train(**params)
    train.infer(**params)
