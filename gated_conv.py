import six
import chainer
import functools
import numpy as np
import nutszebra_chainer
import chainer.links as L
import chainer.functions as F
from collections import defaultdict


class Gated_Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2):
        self.timestep = timestep
        self.pad = timestep - 1
        super(Gated_Conv, self).__init__(
            conv=L.Convolution2D(1, out_channel, (in_channel, timestep), (1, 1), (0, self.pad)),
            conv_f=L.Convolution2D(1, out_channel, (in_channel, timestep), (1, 1), (0, self.pad)),
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
    def remove_pad(x, pad):
        return F.concat(F.split_axis(x, x.data.shape[-1], len(x.data.shape) - 1)[:pad], -1)

    def __call__(self, x, train=False):
        # x: batch,  1, in_channel, input_length
        A = F.relu(self.remove_pad(self.conv(x), -self.pad))
        B = F.sigmoid(self.remove_pad(self.conv_f(x), -self.pad))
        h = A * B
        batch, out_channel, _, input_length = h.shape
        return F.reshape(h, (batch, 1, out_channel, input_length))


class ResBlock(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, timestep=2):
        self.timestep = timestep
        super(ResBlock, self).__init__(
            conv1=Gated_Conv(in_channel, out_channel, timestep),
            conv2=Gated_Conv(out_channel, out_channel, timestep),
        )

    def weight_initialization(self):
        self.conv1.weight_initialization()
        self.conv2.weight_initialization()

    def count_parameters(self):
        return self.conv1.count_parameters() + self.conv2.count_parameters()

    @staticmethod
    def concatenate_zero_pad(x, h_shape, volatile, h_type):
        x = F.swapaxes(x, 1, 2)
        h_shape = (h_shape[0], h_shape[2], h_shape[1], h_shape[3])
        _, x_channel, _, _ = x.data.shape
        batch, h_channel, h_y, h_x = h_shape
        if x_channel == h_channel:
            return F.swapaxes(x, 1, 2)
        pad = chainer.Variable(np.zeros((batch, h_channel - x_channel, h_y, h_x), dtype=np.float32), volatile=volatile)
        if h_type is not np.ndarray:
            pad.to_gpu()
        return F.swapaxes(F.concat((x, pad)), 1, 2)

    def __call__(self, x, train=False):
        h = self.conv1(x, train)
        h = self.conv2(h, train)
        h = h + self.concatenate_zero_pad(x, h.data.shape, h.volatile, type(h.data))
        return h


class Gated_Convolutional_Network(nutszebra_chainer.Model):

    def __init__(self, embed_dimension, category_num):
        super(Gated_Convolutional_Network, self).__init__()
        modules = []
        # register layers
        [self.add_link(*link) for link in modules]
        modules += [('resblock_1', ResBlock(embed_dimension, 128, 3))]
        modules += [('resblock_2', ResBlock(128, 256, 3))]
        modules += [('resblock_3', ResBlock(256, 256, 3))]
        modules += [('resblock_4', ResBlock(256, 256, 3))]
        modules += [('resblock_5', ResBlock(256, 512, 3))]
        modules += [('resblock_6', ResBlock(512, 512, 3))]
        modules += [('resblock_7', ResBlock(512, 512, 3))]
        modules += [('resblock_8', ResBlock(512, 1024, 3))]
        modules += [('resblock_9', ResBlock(1024, 1024, 3))]
        modules += [('resblock_10', ResBlock(1024, 1024, 3))]
        modules += [('gated_conv', Gated_Conv(1024, category_num, 3))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.name = 'Gated_Convolutional_Network_{}_{}'.format(embed_dimension, category_num)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    def __call__(self, x, train=True):
        for i in six.moves.range(1, 10 + 1):
            x = self['resblock_{}'.format(i)](x, train)
        return self.gated_conv(x)

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
