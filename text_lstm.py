# pylint:skip-file
import sys
#sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import resnet
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx):
    """LSTM Cell symbol"""
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    #next_c = mx.sym.element_mask(next_c, mask)
    #next_h = mx.sym.element_mask(next_h, mask)
    return LSTMState(c=next_c, h=next_h)

def lenet(data):
    conv1 = mx.symbol.Convolution(name='conv1',data=data, kernel=(3,3), num_filter=64,pad=(1,1))
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    conv2 = mx.symbol.Convolution(name='conv2',data=relu1, kernel=(3,3), num_filter=64,pad=(1,1))
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    bn1 = mx.sym.BatchNorm(name='batchnorm1',data=relu2, fix_gamma=False)
    pool1 = mx.symbol.Pooling(data=bn1, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv3 = mx.symbol.Convolution(name='conv3',data=pool1, kernel=(3,3), num_filter=128,pad=(1,1))
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(name='conv4',data=relu3, kernel=(3,3), num_filter=128,pad=(1,1))
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    bn2 = mx.sym.BatchNorm(name='batchnorm2',data=relu4, fix_gamma=False)
    pool2 = mx.symbol.Pooling(data=bn2, pool_type="max", kernel=(2,2), stride=(2, 2))

    conv5 = mx.symbol.Convolution(name='conv5',data=pool2, kernel=(3,3), num_filter=256,pad=(1,1))
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    conv6 = mx.symbol.Convolution(name='conv6',data=relu5, kernel=(3,3), num_filter=256,pad=(1,1))
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu")
    bn3 = mx.sym.BatchNorm(data=relu6, fix_gamma=False)
    pool3 = mx.symbol.Pooling(data=relu6, pool_type="max", kernel=(2,2), stride=(2, 2))
    conv7 = mx.symbol.Convolution(name='conv7',data=pool3, kernel=(1,1), num_filter=512,pad=(0,0))
    return conv7

def lstm_unroll(num_lstm_layer, seq_len,
                num_hidden, num_label,dropout=0):
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    conv=lenet(data)
    column_features = mx.sym.SliceChannel(data=data, num_outputs=seq_len,axis=3, squeeze_axis=1)

    hidden_all = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i)
            hidden = next_state.h
            last_states[i] = next_state
        if dropout > 0:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_concat, num_hidden=2369)
    print (pred.infer_shape(data=(32,3,32,320)))
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    sm = mx.sym.WarpCTC(data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm
def bi_lstm_unroll(seq_len,
                num_hidden, num_label,dropout=0):
    last_states = []
    last_states.append(LSTMState(c = mx.sym.Variable("l0_init_c"), h = mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c = mx.sym.Variable("l1_init_c"), h = mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    assert(len(last_states) == 2)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')

    from importlib import import_module
    resnet = import_module('resnet')
    conv=resnet.get_symbol(2, 18, '3,32,'+str(seq_len*8))

    print ('seq_len : ',seq_len)
    column_features = mx.sym.SliceChannel(data=conv, num_outputs=seq_len,axis=3, squeeze_axis=1)
    hidden_all = []
    forward_hidden = []
    for seqidx in range(seq_len):
        hidden =mx.sym.Flatten(data=column_features[seqidx])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0)
        hidden = next_state.h
        last_states[0] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
  
        forward_hidden.append(hidden)


    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden =mx.sym.Flatten(data=column_features[k])
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1)
        hidden = next_state.h
        last_states[1] = next_state
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        backward_hidden.insert(0, hidden)
        
    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(name='fc1',data=hidden_concat, num_hidden=98)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    label = mx.sym.Cast(data = label, dtype = 'int32')
    #does Warp-CTC support bucketing?
    sm = mx.sym.WarpCTC(name='ctc-loss',data=pred, label=label, label_length = num_label, input_length = seq_len)
    return sm
