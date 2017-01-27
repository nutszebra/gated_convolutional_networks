import sys
sys.path.append('./trainer')
import six
import chainer
import functools
import numpy as np
import nutszebra_chainer
import chainer.links as L
import chainer.functions as F


class DoNothing(object):

    def __call__(self, x):
        return x


class Gated_Linear_Unit(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2, activation=F.relu):
        self.timestep = timestep
        self.pad = timestep - 1
        self.activation = activation
        super(Gated_Linear_Unit, self).__init__(
            conv=L.Convolution2D(1, out_channel, (in_channel, timestep), 1, 0),
            conv_f=L.Convolution2D(1, out_channel, (in_channel, timestep), 1, 0),
        )

    @staticmethod
    def _count_parameters(link):
        return functools.reduce(lambda a, b: a * b, link.W.data.shape)

    def count_parameters(self):
        count = 0
        for name in ['conv', 'conv_f']:
            count += self._count_parameters(self[name])
        return count

    def _weight_initialization(self, name):
        self[name].W.data = self.weight_relu_initialization(self[name])
        self[name].b.data = self.bias_initialization(self[name], constant=0)

    def weight_initialization(self):
        for name in ['conv', 'conv_f']:
            self._weight_initialization(name)

    @staticmethod
    def add_zero_pad(x, pad, axis, front=True, dtype=np.float32):
        if pad < 1:
            return x
        sizes = list(x.data.shape)
        sizes[axis] = pad
        pad_mat = chainer.Variable(np.zeros(sizes, dtype=dtype), volatile=x.volatile)
        if not type(x.data) == np.ndarray:
            pad_mat.to_gpu()
        if front:
            return F.concat((pad_mat, x), axis=axis)
        else:
            return F.concat((x, pad_mat), axis=axis)

    def __call__(self, x, train=False):
        # x: batch,  1, in_channel, input_length
        A = self.activation(self.conv(self.add_zero_pad(x, self.pad, 3)))
        B = F.sigmoid(self.conv_f(self.add_zero_pad(x, self.pad, 3)))
        h = A * B
        batch, out_channel, _, input_length = h.shape
        return F.reshape(h, (batch, 1, out_channel, input_length))


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2):
        self.timestep = timestep
        super(ResBlock, self).__init__(
            conv1=Gated_Linear_Unit(in_channel, out_channel, timestep),
            conv2=Gated_Linear_Unit(out_channel, out_channel, timestep),
        )

    def weight_initialization(self):
        self.conv1.weight_initialization()
        self.conv2.weight_initialization()

    def count_parameters(self):
        return self.conv1.count_parameters() + self.conv2.count_parameters()

    def __call__(self, x, train=False):
        h = self.conv1(x, train)
        h = self.conv2(h, train)
        diff_channel = h.data.shape[2] - x.data.shape[2]
        return h + Gated_Linear_Unit.add_zero_pad(x, diff_channel, 2)


class Gated_Convolutional_Network(nutszebra_chainer.Model):

    def __init__(self, embed_dimension, category_num):
        super(Gated_Convolutional_Network, self).__init__()
        modules = []
        # register layers
        [self.add_link(*link) for link in modules]
        modules += [('resblock_1', ResBlock(embed_dimension, 16, 5))]
        modules += [('resblock_2', ResBlock(16, 16, 5))]
        modules += [('resblock_3', ResBlock(16, 16, 5))]
        modules += [('resblock_4', ResBlock(16, 16, 5))]
        modules += [('resblock_5', ResBlock(16, 32, 5))]
        modules += [('gated_conv', Gated_Linear_Unit(32, category_num, 5, DoNothing()))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.embed_dimension = embed_dimension
        self.category_num = category_num
        self.name = 'Gated_Convolutional_Network_{}_{}'.format(embed_dimension, category_num)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=True):
        for i in six.moves.range(1, 5 + 1):
            x = self['resblock_{}'.format(i)](x, train)
        batch = x.data.shape[0]
        return F.reshape(self.gated_conv(x), (batch, self.category_num, -1))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss
