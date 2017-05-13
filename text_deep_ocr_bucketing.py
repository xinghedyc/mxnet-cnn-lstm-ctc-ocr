# pylint: disable=C0111,too-many-arguments,too-many-instance-attributes,too-many-locals,redefined-outer-name,fixme
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random
#sys.path.insert(0, "../../python")
import numpy as np
import mxnet as mx

from text_lstm import lstm_unroll,bi_lstm_unroll
from io import BytesIO
from captcha.image import ImageCaptcha
import cv2, random
from text_bucketing_iter import TextIter



def get_label(buf):
    ret = np.zeros(4)
    for i in range(len(buf)):
        ret[i] = 1 + int(buf[i])
    if len(buf) == 3:
        ret[3] = 0
    return ret


BATCH_SIZE = 20
def ctc_label(p):
    ret = []
    p1 = [0] + p
    for i in range(len(p)):
        c1 = p1[i]
        c2 = p1[i+1]
        if c2 == 0 or c2 == c1:
            continue
        ret.append(c2)
    return ret

def remove_blank(l):
    ret = []
    for i in range(len(l)):
        if l[i] == 0:
            break
        ret.append(l[i])
    return ret

def Accuracy(label, pred):
    global BATCH_SIZE
    global SEQ_LENGTH
    #print (label,pred)
    hit = 0.
    total = 0.
    for i in range(BATCH_SIZE):
        l = remove_blank(label[i])
        p = []
        #print ("pred length : ", len(pred))
        for k in range(len(pred)/BATCH_SIZE):
            p.append(np.argmax(pred[k * BATCH_SIZE + i]))
        p = ctc_label(p)
        #print p,l
        if len(p) == len(l):
            match = True
            for k in range(len(p)):
                if p[k] != int(l[k]):
                    match = False
                    break
            if match:
                hit += 1.0
        total += 1.0
    return hit / total

if __name__ == '__main__':
    num_hidden = 256
    num_lstm_layer = 2
    num_epoch = 200
    num_label = 10
    contexts = [mx.gpu(0)]
    def sym_gen(seq_len):
        return bi_lstm_unroll(seq_len,
                           num_hidden=num_hidden,
                           num_label = num_label,dropout=0.75),('data','l0_init_c','l1_init_c','l0_init_h','l1_init_h'), ('label',)

    init_c = [('l%d_init_c'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (BATCH_SIZE, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h
    path='crop_icdar2013_train.lst'
    path_test='crop_icdar2013_val.lst'
    data_root='/cache/icdar2013_word'
    test_root='/cache/icdar2013_word'
    buckets=[4*i for i in range(1,num_label+1) ]
    data_train=TextIter(path,data_root, BATCH_SIZE, init_states,num_label,buckets=buckets)
    data_val=TextIter(path_test, test_root, BATCH_SIZE,init_states,num_label,buckets=buckets)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = 40,
        context             = contexts)    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)
    mon = mx.mon.Monitor(100, norm_stat)     
    prefix='model/icdar2013'

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = mx.metric.np(Accuracy),
        #kvstore             = args.kv_store,
        optimizer           = 'sgd',
        optimizer_params    = { 'learning_rate': 0.005,
                                'momentum': 0.9,
                                'wd': 0.0005 },
        kvstore             = 'device',
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        num_epoch           = num_epoch,
        epoch_end_callback =mx.callback.do_checkpoint(prefix),
        batch_end_callback  = mx.callback.Speedometer(BATCH_SIZE, 50),
        begin_epoch=0
        )
