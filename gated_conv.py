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


class Conv_For_GLU(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=3):
        super(Conv_For_GLU, self).__init__(
            conv=L.Convolution2D(1, out_channel, (in_channel, timestep), 1, 0),
        )
        self.pad = timestep - 1

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

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
        return self.conv(self.add_zero_pad(x, self.pad, 3))


class Gated_Unit(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2, activation=F.relu):
        super(Gated_Unit, self).__init__()
        modules = []
        modules += [('conv', Conv_For_GLU(in_channel, out_channel, timestep))]
        modules += [('conv_f', Conv_For_GLU(in_channel, out_channel, timestep))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.pad = timestep - 1
        self.activation = activation

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        # x: batch,  1, in_channel, input_length
        A = self.activation(self.conv(x, train))
        B = F.sigmoid(self.conv_f(x, train))
        h = A * B
        batch, out_channel, _, input_length = h.shape
        return F.reshape(h, (batch, 1, out_channel, input_length))


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2):
        super(ResBlock, self).__init__()
        modules = []
        modules += [('conv1', Gated_Unit(in_channel, out_channel, timestep))]
        modules += [('conv2', Gated_Unit(out_channel, out_channel, timestep))]
        modules += [('conv3', Gated_Unit(out_channel, out_channel, timestep))]
        modules += [('conv4', Gated_Unit(out_channel, out_channel, timestep))]
        modules += [('conv5', Gated_Unit(out_channel, out_channel, timestep))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=False):
        h = x
        for i in six.moves.range(1, 5 + 1):
            h = self['conv{}'.format(i)](h, train)
        diff_channel = h.data.shape[2] - x.data.shape[2]
        return h + Conv_For_GLU.add_zero_pad(x, diff_channel, 2)


class Gated_Convolutional_Network(nutszebra_chainer.Model):

    def __init__(self, embed_dimension, category_num):
        super(Gated_Convolutional_Network, self).__init__()
        modules = []
        # register layers
        [self.add_link(*link) for link in modules]
        modules += [('resblock_1', ResBlock(embed_dimension, 4, 4))]
        modules += [('resblock_2', ResBlock(4, 4, 4))]
        # modules += [('gated_conv', Gated_Unit(32, category_num, 3, F.tanh))]
        modules += [('conv', Conv_For_GLU(4, category_num, 4))]
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
        for i in six.moves.range(1, 2 + 1):
            x = self['resblock_{}'.format(i)](x, train)
        batch = x.data.shape[0]
        return F.reshape(self.conv(x), (batch, self.category_num, -1))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss
