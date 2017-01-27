import argparse
import random
import itertools
import chainer
import numpy as np
import gated_conv
import nutszebra_sampling
from six.moves import range
from six.moves import zip
from nutszebra_utility import Utility as utility
from chainer import optimizers
import nutszebra_basic_print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='bit inversion')
    parser.add_argument('--gpu', '-g', type=int,
                        default=-1,
                        help='-1 means cpu mode, put gpu id here')
    parser.add_argument('--batch', '-b', type=int,
                        default=128,
                        help='batch number')
    parser.add_argument('--lr', '-lr', type=float,
                        default=0.1,
                        help='learning rate')
    parser.add_argument('--epoch', '-e', type=int,
                        default=200,
                        help='total epoch')
    print('learng bit inversion')
    print(' example: 0100100101 -> 0111000110')
    args = parser.parse_args().__dict__
    g = args.pop('gpu')
    batch = args.pop('batch')
    lr = args.pop('lr')
    epoch = args.pop('epoch')
    print('arguments')
    print(' gpu: {}'.format(g))
    print(' batch: {}'.format(batch))
    print(' lr: {}'.format(lr))
    print(' epoch: {}'.format(epoch))

    print('creat dataset')
    data = []
    for bits in itertools.product(*tuple([range(2) for _ in range(15)])):
        y = [bits[0]]
        for bit in bits[1:]:
            y.append(y[-1] ^ bit)
        data.append((np.array(bits, dtype=np.float32), np.array(y, dtype=np.int32)))
    random.shuffle(data)
    train_x, train_y = zip(*data[:-1000])
    test_x, test_y = zip(*data[-1000:])
    print(' done')

    print('define model')
    model = gated_conv.Gated_Convolutional_Network(1, 2)
    print(' done')

    print('weight initialization')
    model.weight_initialization()
    print(' done')

    if g >= 0:
        print('gpu mode')
        model.to_gpu(g)

    sample = nutszebra_sampling.Sampling()

    epoch_bar = utility.create_progressbar(epoch, desc='epoch', start=1)

    print('set optimizier')
    optimizer = optimizers.MomentumSGD(lr, 0.9)
    weight_decay = chainer.optimizer.WeightDecay(1.0e-4)
    clip = chainer.optimizer.GradientClipping(0.1)
    optimizer.setup(model)
    optimizer.add_hook(weight_decay)
    optimizer.add_hook(clip)
    print(' lr: {}'.format(lr))
    print(' weight decay: {}'.format(1.0e-4))
    print(' gradient clipping: {}'.format(0.1))

    print('start learning')
    for i in epoch_bar:
        if i in [int(epoch * 0.5), int(epoch * 0.75)]:
            optimizer.lr /= 10
        train_bar = utility.create_progressbar(int(len(train_x) / batch), desc='train', stride=1, start=1)
        accum_loss = 0
        for _ in train_bar:
            model.cleargrads()
            indices = sample.pick_random_permutation(batch, len(train_x))
            x = []
            t = []
            for ind in indices:
                x.append([[train_x[ind]]])
                t.append(train_y[ind])
            x = model.prepare_input(x, dtype=np.float32, volatile=False)
            t = model.prepare_input(t, dtype=np.int32, volatile=False)
            y = model(x, train=True)
            loss = model.calc_loss(y, t)
            loss.backward()
            optimizer.update()
            loss.to_cpu()
            accum_loss += loss.data
        print('epoch {}: train-loss, {}'.format(i, accum_loss / len(train_x)))
        test_bar = utility.create_progressbar(len(test_x), desc='test', stride=batch, start=0)
        accuracy = 0
        for ii in test_bar:
            x = np.reshape(test_y[ii:ii + batch], (batch, 1, 1, -1))
            t = test_y[ii:ii + batch]
            x = model.prepare_input(x, dtype=np.float32, volatile=True)
            y = np.argmax(model(x, train=False).data, 1)
            for i in range(batch):
                accuracy += int(np.all(t[i] == y[i]))
        print('epoch {}: test-accuracy {}'.format(i, float(accuracy) / len(test_x)))
