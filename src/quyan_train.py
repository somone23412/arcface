# THIS FILE IS FOR EXPERIMENTS, USE train_softmax.py FOR NORMAL TRAINING.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
from data import FaceImageIter
from data import FaceImageIterList
## My DataIter
from data_iter_qsr import MultiResolutionFaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
from noise_sgd import NoiseSGD
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet 
import fmobilenetv2
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
#import lfw
import verification
import sklearn
sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
import center_loss
import time
import copy

logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None


class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    self.count+=1
    if args.loss_type>=2 and args.loss_type<=7 and args.margin_verbose>0:
      if self.count%args.ctx_num==0:
        mbatch = self.count//args.ctx_num
        _verbose = args.margin_verbose
        if mbatch==1 or mbatch%_verbose==0:
          a = 0.0
          b = 0.0
          if len(preds)>=4:
            a = preds[-2].asnumpy()[0]
            b = preds[-1].asnumpy()[0]
          elif len(preds)==3:
            a = preds[-1].asnumpy()[0]
            b = a
          print('[%d][MARGIN]%f,%f'%(mbatch,a,b))
    if args.logits_verbose>0:
      if self.count%args.ctx_num==0:
        mbatch = self.count//args.ctx_num
        _verbose = args.logits_verbose
        if mbatch==1 or mbatch%_verbose==0:
          a = 0.0
          b = 0.0
          if len(preds)>=3:
            v = preds[-1].asnumpy()
            v = np.sort(v)
            num = len(v)//10
            a = np.mean(v[0:num])
            b = np.mean(v[-1*num:])
          print('[LOGITS] %d,%f,%f'%(mbatch,a,b))
    #loss = preds[2].asnumpy()[0]
    #if len(self.losses)==20:
    #  print('ce loss', sum(self.losses)/len(self.losses))
    #  self.losses = []
    #self.losses.append(loss)
    preds = [preds[1]] #use softmax output
    for label, pred_label in zip(labels, preds):
        #print(pred_label)
        #print(label.shape, pred_label.shape)
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        label = label.asnumpy()
        if label.ndim==2:
          label = label[:,0]
        label = label.astype('int32').flatten()
        #print(label)
        #print('label',label)
        #print('pred_label', pred_label)
        assert label.shape==pred_label.shape
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)

class LossValueMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(LossValueMetric, self).__init__(
        'lossvalue', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []

  def update(self, labels, preds):
    loss = preds[-1].asnumpy()[0]
    self.sum_metric += loss
    self.num_inst += 1.0
    gt_label = preds[-2].asnumpy()
    #print(gt_label)

def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--noise-sgd', type=float, default=0.0, help='')
  parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--per-batch-size', type=int, default=128, help='batch size in each context')
  parser.add_argument('--margin-m', type=float, default=0.5, help='')
  parser.add_argument('--margin-s', type=float, default=64.0, help='')
  parser.add_argument('--margin-a', type=float, default=0.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--margin-verbose', type=int, default=0, help='')
  parser.add_argument('--logits-verbose', type=int, default=0, help='')
  parser.add_argument('--c2c-threshold', type=float, default=0.0, help='')
  parser.add_argument('--c2c-mode', type=int, default=-10, help='')
  parser.add_argument('--output-c2c', type=int, default=0, help='')
  parser.add_argument('--train-limit', type=int, default=0, help='')
  parser.add_argument('--margin', type=int, default=4, help='')
  parser.add_argument('--beta', type=float, default=1000., help='')
  parser.add_argument('--beta-min', type=float, default=5., help='')
  parser.add_argument('--beta-freeze', type=int, default=0, help='')
  parser.add_argument('--gamma', type=float, default=0.12, help='')
  parser.add_argument('--power', type=float, default=1.0, help='')
  parser.add_argument('--scale', type=float, default=0.9993, help='')
  parser.add_argument('--center-alpha', type=float, default=0.5, help='')
  parser.add_argument('--center-scale', type=float, default=0.003, help='')
  parser.add_argument('--images-per-identity', type=int, default=0, help='')
  parser.add_argument('--triplet-bag-size', type=int, default=3600, help='')
  parser.add_argument('--triplet-alpha', type=float, default=0.3, help='')
  parser.add_argument('--triplet-max-ap', type=float, default=0.0, help='')
  parser.add_argument('--verbose', type=int, default=2000, help='')
  parser.add_argument('--loss-type', type=int, default=4, help='')
  parser.add_argument('--incay', type=float, default=0.0, help='feature incay')
  parser.add_argument('--use-deformable', type=int, default=0, help='')
  parser.add_argument('--rand-mirror', type=int, default=1, help='')
  parser.add_argument('--patch', type=str, default='0_0_96_112_0',help='')
  parser.add_argument('--lr-steps', type=str, default='', help='')
  parser.add_argument('--max-steps', type=int, default=0, help='')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--color', type=int, default=0, help='color jittering aug')
  parser.add_argument('--images-filter', type=int, default=0, help='minimum images per identity filter')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30,cplfw,calfw', help='')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  margin_symbols = []
  if args.network[0]=='r':
    print('init resnet', args.num_layers)
    embedding = fresnet.get_symbol(args.emb_size, args.num_layers, 
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act)
  all_label = mx.symbol.Variable('softmax_label')
  if not args.output_c2c:
    gt_label = all_label
  else:
    gt_label = mx.symbol.slice_axis(all_label, axis=1, begin=0, end=1)
    gt_label = mx.symbol.reshape(gt_label, (args.per_batch_size,))
    c2c_label = mx.symbol.slice_axis(all_label, axis=1, begin=1, end=2)
    c2c_label = mx.symbol.reshape(c2c_label, (args.per_batch_size,))
  assert args.loss_type>=0
  extra_loss = None
  ## arc face
  if args.loss_type==4:
    s = args.margin_s
    m = args.margin_m
    assert s>0.0
    assert m>=0.0
    assert m<(math.pi/2)
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance')
    nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*s
    fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
    zy = mx.sym.pick(fc7, gt_label, axis=1)
    cos_t = zy/s
    if args.margin_verbose>0:
      margin_symbols.append(mx.symbol.mean(cos_t))
    if args.output_c2c==0:
      cos_m = math.cos(m)
      sin_m = math.sin(m)
      mm = math.sin(math.pi-m)*m
      #threshold = 0.0
      threshold = math.cos(math.pi-m)
      if args.easy_margin:
        cond = mx.symbol.Activation(data=cos_t, act_type='relu')
      else:
        cond_v = cos_t - threshold
        cond = mx.symbol.Activation(data=cond_v, act_type='relu')
      body = cos_t*cos_t
      body = 1.0-body
      sin_t = mx.sym.sqrt(body)
      new_zy = cos_t*cos_m
      b = sin_t*sin_m
      new_zy = new_zy - b
      new_zy = new_zy*s
      if args.easy_margin:
        zy_keep = zy
      else:
        zy_keep = zy - s*mm
      new_zy = mx.sym.where(cond, new_zy, zy_keep)
    else:
      #set c2c as cosm^2 in data.py
      cos_m = mx.sym.sqrt(c2c_label)
      sin_m = 1.0-c2c_label
      sin_m = mx.sym.sqrt(sin_m)
      body = cos_t*cos_t
      body = 1.0-body
      sin_t = mx.sym.sqrt(body)
      new_zy = cos_t*cos_m
      b = sin_t*sin_m
      new_zy = new_zy - b
      new_zy = new_zy*s

    if args.margin_verbose>0:
      new_cos_t = new_zy/s
      margin_symbols.append(mx.symbol.mean(new_cos_t))
    diff = new_zy - zy
    diff = mx.sym.expand_dims(diff, 1)
    gt_one_hot = mx.sym.one_hot(gt_label, depth = args.num_classes, on_value = 1.0, off_value = 0.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)
    fc7 = fc7+body
  else:
    #embedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')*float(args.loss_type)
    embedding = embedding * 5
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0)
    _weight = mx.symbol.L2Normalization(_weight, mode='instance') * 2
    fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                          weight = _weight,
                          beta=args.beta, margin=args.margin, scale=args.scale,
                          beta_min=args.beta_min, verbose=100, name='fc7')

    #fc7 = mx.sym.Custom(data=embedding, label=gt_label, weight=_weight, num_hidden=args.num_classes,
    #                       beta=args.beta, margin=args.margin, scale=args.scale,
    #                       op_type='ASoftmax', name='fc7')
  out_list = [mx.symbol.BlockGrad(embedding)]
  # out_list = [embedding]
  softmax = None
  if args.loss_type<10:
    softmax = mx.symbol.SoftmaxOutput(data=fc7, label = gt_label, name='softmax', normalization='valid')
    out_list.append(softmax)
    if args.logits_verbose>0:
      logits = mx.symbol.softmax(data = fc7)
      logits = mx.sym.pick(logits, gt_label, axis=1)
      margin_symbols.append(logits)
  if softmax is None:
    out_list.append(mx.sym.BlockGrad(gt_label))
  if extra_loss is not None:
    out_list.append(extra_loss)
  for _sym in margin_symbols:
    _sym = mx.sym.BlockGrad(_sym)
    out_list.append(_sym)
  # out = mx.symbol.Group(out_list)
  return embedding, out_list

def get_HR_symbol(args, arg_params, aux_params):
  ## input embedding from LR network
  LR_embedding = mx.symbol.Variable('LR_embedding')
  HR_embedding, out_list = get_symbol(args, arg_params, aux_params)
  ## generator, bottleneck network
  ## ...
  G_bn1 = mx.sym.BatchNorm(HR_embedding, fix_gamma=False, eps=2e-5, momentum=0.9, name='G_bn1')
  G_relu1 = mx.sym.LeakyReLU(G_bn1, act_type='prelu', name='G_relu1')
  G_drop1 = mx.sym.Dropout(G_relu1, p=0.5, name='G_drop1')

  G_fc2 = mx.sym.FullyConnected(G_drop1, name='G_fc2', num_hidden=64)
  G_bn2 = mx.sym.BatchNorm(G_fc2, fix_gamma=False, eps=2e-5, momentum=0.9, name='G_bn2')
  G_relu2 = mx.sym.LeakyReLU(G_bn2, act_type='prelu', name='G_relu2')
  G_drop2 = mx.sym.Dropout(G_relu2, p=0.5, name='G_drop2')

  G_embedding = mx.sym.FullyConnected(G_drop1, name='G_fc2', num_hidden=512)
  # G_embedding = HR_embedding
  # diff = embedding_1 - embedding_2
  # diff = diff * diff
  # cos_sim_diff
  # diff = mx.symbol.sum(diff, axis = 1, keepdims=True) / (args.emb_size)
  diff = 1 - mx.symbol.sum((G_embedding * LR_embedding), axis = 1, keepdims=True) / mx.symbol.sqrt(
    mx.symbol.sum((G_embedding * G_embedding), axis = 1, keepdims=True) * mx.symbol.sum((LR_embedding * LR_embedding), axis = 1, keepdims=True)
  )
  diff = 0.00005 * diff * diff 
  loss_diff = mx.sym.MakeLoss(diff)
  out_list.append(loss_diff)
  print("HR sym out = ", out_list)
  out = mx.symbol.Group(out_list)
  return out
  
def get_LR_symbol(args, arg_params, aux_params):
  LR_embedding, out_list = get_symbol(args, arg_params, aux_params)
  print("LR sym out = ", out_list)
  out = mx.symbol.Group(out_list)
  return out

def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size==0:
      args.per_batch_size = 128
      if args.loss_type==10:
        args.per_batch_size = 256
    args.batch_size = args.per_batch_size*args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3
    ppatch = [int(x) for x in args.patch.split('_')]
    assert len(ppatch)==5


    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    if args.loss_type!=12 and args.loss_type!=13:
      assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    args.use_val = False
    path_imgrec = None
    path_imglist = None
    val_rec = None
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)

    assert(args.num_classes>0)
    print('num_classes', args.num_classes)
    args.coco_scale = 0.5*math.log(float(args.num_classes-1))+3

    #path_imglist = "/raid5data/dplearn/MS-Celeb-Aligned/lst2"
    path_imgrec = os.path.join(data_dir, "train.rec")
    val_rec = os.path.join(data_dir, "val.rec")
    if os.path.exists(val_rec) and args.loss_type<10:
      args.use_val = True
    else:
      val_rec = None
    #args.use_val = False

    if args.loss_type==1 and args.num_classes>20000:
      args.beta_freeze = 5000
      args.gamma = 0.06

    assert args.images_per_identity==0

    print('Called with argument:', args)

    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None




    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      LR_sym = get_LR_symbol(args, arg_params, aux_params)
      HR_sym = get_HR_symbol(args, arg_params, aux_params)
    else:
      vec = args.pretrained.split(',')
      print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      LR_sym = get_LR_symbol(args, arg_params, aux_params)
      HR_sym = get_HR_symbol(args, arg_params, aux_params)
    
    
    data_extra = None
    hard_mining = False
    triplet_params = None
    coco_mode = False

    label_name = 'softmax_label'
    label_shape = (args.batch_size,)
    if args.output_c2c:
      label_shape = (args.batch_size,2)

    # print('line 398, sym.get_internals()[\'fc1_output\'] = ', sym.get_internals()['fc1_output'])
    
    model_H = mx.mod.Module(
      context       = ctx,
      symbol        = HR_sym,
      label_names    = ('softmax_label', 'LR_embedding'),
    )
    
    model_L = mx.mod.Module(
      context       = ctx,
      symbol        = LR_sym,
    )
    
    
    val_dataiter = None


    train_dataiter = MultiResolutionFaceImageIter(
        batch_size           = args.batch_size,
        data_shape           = data_shape,
        path_imgrec          = path_imgrec,
        shuffle              = True,
        rand_mirror          = args.rand_mirror,
        mean                 = mean,
        cutoff               = args.cutoff,
        color_jittering      = args.color,
        images_filter        = args.images_filter,
    )

    if args.loss_type<10:
      _metric = AccMetric()
    else:
      _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0]=='r':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    if args.noise_sgd>0.0:
      print('use noise sgd')
      opt = NoiseSGD(scale = args.noise_sgd, learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    else:
      opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      path = os.path.join(data_dir,name+".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)



    def ver_test(nbatch):
      results = []
      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model_H, args.batch_size, 10, data_extra, label_shape)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results


    def val_test():
      acc = AccMetric()
      val_metric = mx.metric.create(acc)
      val_metric.reset()
      val_dataiter.reset()
      for i, eval_batch in enumerate(val_dataiter):
        model_H.forward(eval_batch, is_train=False)
        model_H.update_metric(val_metric, eval_batch.label)
      acc_value = val_metric.get_name_value()[0][1]
      print('VACC: %f'%(acc_value))

    ## modified by quyan @ 2019.9.19 : expand highest list
    # highest_acc = [0.0, 0.0]  #lfw and target
    highest_acc = [0.0, 0.0, 0.0, 0.0, 0.0]
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [40000, 60000, 80000]
      if args.loss_type>=1 and args.loss_type<=7:
        lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==args.beta_freeze+_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      if mbatch%1000==0:
        print('lr-batch-epoch:',opt.lr,param.nbatch,param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        acc_list = ver_test(mbatch)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        if len(acc_list)>0:
            score = {}
            score['lfw_score'] = acc_list[0]
            score['cfp_score'] = acc_list[1]
            score['agedb_score'] = acc_list[2]
            score['cplfw_score'] = acc_list[3]
            score['calfw_score'] = acc_list[4]
            print('score=', score)
            if score['lfw_score'] > highest_acc[0]:
                highest_acc[0] = score['lfw_score']
                if score['lfw_score'] >= 0.99:
                    do_save = True
                    print('[attention] lfw_score >= 0.99, save = True, epoch = ', msave)
            if score['cfp_score'] > highest_acc[1]:
                highest_acc[1] = score['cfp_score']
                if score['cfp_score'] > 0.94:
                    do_save = True
                    
                    print('[attention] cfp_score >= 0.94, save = True, epoch = ', msave)
            if score['agedb_score'] > highest_acc[2]:
                highest_acc[2] = score['agedb_score']
                if score['agedb_score'] > 0.93:
                    do_save = True
                    print('[attention] agedb_score >= 0.93, save = True, epoch = ', msave)
                        
            if score['cplfw_score'] > highest_acc[3]:
                highest_acc[3] = score['cplfw_score']
                if score['cplfw_score'] > 0.85:
                    do_save = True
                    print('[attention] cplfw_score >= 0.85, save = True, epoch = ', msave)
            if score['calfw_score'] > highest_acc[4]:
                highest_acc[4] = score['calfw_score']
                if score['calfw_score'] > 0.9:
                    do_save = True
                    print('[attention] calfw_score >= 0.9, save = True, epoch = ', msave)
        ## modified by quyan @ 2019.9.12 : always save
        do_save = True
        if args.ckpt == 0:
            do_save = False
        elif args.ckpt > 1:
            do_save = True
        arg, aux = model_H.get_params()
        print('saving', 0)
        mx.model.save_checkpoint(prefix, 0, model_H.symbol, arg, aux)
        if do_save:
            print('saving', msave)
            mx.model.save_checkpoint(prefix, msave, model_H.symbol, arg, aux)
        print(
            '[%d]score_highest: lfw: %1.5f cfp: %1.5f agedb: %1.5f cplfw: %1.5f calfw: %1.5f'
            % (mbatch, highest_acc[0], highest_acc[1], highest_acc[2], highest_acc[3], highest_acc[4])
        )
      if mbatch<=args.beta_freeze:
        _beta = args.beta
      else:
        move = max(0, mbatch-args.beta_freeze)
        _beta = max(args.beta_min, args.beta*math.pow(1+args.gamma*move, -1.0*args.power))
      #print('beta', _beta)
      os.environ['BETA'] = str(_beta)
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    #epoch_cb = mx.callback.do_checkpoint(prefix, 1)
    epoch_cb = None



    #def _epoch_callback(epoch, sym, arg_params, aux_params):
    #  print('epoch-end', epoch)

    ## modified by quyan @ 2019.9.19 : change to for loop
    # model.fit(train_dataiter,
        # begin_epoch        = begin_epoch,
        # num_epoch          = end_epoch,
        # eval_data          = val_dataiter,
        # eval_metric        = eval_metrics,
        # kvstore            = 'device',
        # optimizer          = opt,
        # #optimizer_params   = optimizer_params,
        # initializer        = initializer,
        # arg_params         = arg_params,
        # aux_params         = aux_params,
        # allow_missing      = True,
        # batch_end_callback = _batch_callback,
        # epoch_end_callback = epoch_cb )
    model_H.bind(data_shapes=train_dataiter.provide_data, 
        label_shapes=[train_dataiter.provide_label[0], ('LR_embedding', (args.batch_size, args.emb_size))],
        for_training=True, force_rebind=False)
    model_H.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
        allow_missing=True, force_init=False)
    model_H.init_optimizer(kvstore='device', optimizer=opt)
    
    model_L.bind(data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label,
        for_training=True, force_rebind=False)
    model_L.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
        allow_missing=True, force_init=False)
    model_L.init_optimizer(kvstore='device', optimizer=opt)
    
    if not isinstance(eval_metrics, mx.model.metric.EvalMetric):
        eval_metrics = mx.model.metric.create(eval_metrics)
    epoch_eval_metric = copy.deepcopy(eval_metrics)

    ################################################################################
    # training loop
    ################################################################################
    for epoch in range(begin_epoch, end_epoch):
        tic = time.time()
        eval_metrics.reset()
        epoch_eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_dataiter)
        end_of_batch = False
        next_data, lr_next_data, next_label, next_num = next(data_iter)
        next_data_batch = mx.io.DataBatch([next_data], [next_label], next_num)
        lr_next_data_batch = mx.io.DataBatch([lr_next_data], [next_label], next_num)
        while not end_of_batch:
            data_batch = next_data_batch
            lr_data_batch = lr_next_data_batch
            
            model_L.forward_backward(lr_data_batch)
            embedding_L = mx.nd.array(model_L.get_outputs()[0].asnumpy())
            model_L.update()
            
            model_H.forward_backward(mx.io.DataBatch([next_data], [next_label, embedding_L], next_num))
            model_H.update()
            HR_outputs = model_H.get_outputs()
            
            ## cossim shape = [batchsize, 1, 1]
            softmax_loss = HR_outputs[1].asnumpy()
            softmax_loss = softmax_loss.sum(axis = 0)/ len(softmax_loss)
            cossim_loss = HR_outputs[2].asnumpy()
            cossim_loss = cossim_loss.sum(axis = 0)/ len(cossim_loss)
            print('softmax_loss=', softmax_loss[0], 'cossim_loss=', cossim_loss[0])

            if isinstance(data_batch, list):
                model_H.update_metric(eval_metrics,
                                   [db.label for db in data_batch],
                                   pre_sliced=True)
                model_H.update_metric(epoch_eval_metric,
                                   [db.label for db in data_batch],
                                   pre_sliced=True)
            else:
                model_H.update_metric(eval_metrics, data_batch.label)
                model_H.update_metric(epoch_eval_metric, data_batch.label)

            try:
                # pre fetch next batch
                next_data, lr_next_data, next_label, next_num = next(data_iter)
                next_data_batch = mx.io.DataBatch([next_data], [next_label], next_num)
                lr_next_data_batch = mx.io.DataBatch([lr_next_data], [next_label], next_num)
                model_H.prepare(next_data_batch, sparse_row_id_fn=None)
                model_L.prepare(lr_next_data_batch, sparse_row_id_fn=None)
            except StopIteration:
                end_of_batch = True

            if end_of_batch:
                eval_name_vals = epoch_eval_metric.get_name_value()

            batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                             eval_metric=eval_metrics,
                                             locals=locals())
            _batch_callback(batch_end_params)
            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_name_vals:
            model_H.logger.info('model_H Epoch[%d] Train-%s=%f', epoch, name, val)
        toc = time.time()
        model_H.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
        
        # sync aux params across devices
        arg_params, aux_params = model_H.get_params()
        model_H.set_params(arg_params, aux_params)
        model_L.set_params(arg_params, aux_params)

        train_dataiter.reset()

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

