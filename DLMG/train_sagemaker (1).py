import argparse
import logging
import os
from time import time

import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    return parser.parse_args()

def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


def get_data(batch_size):
    mnist_train = datasets.MNIST(train=True)
    train_data = gluon.data.DataLoader(mnist_train.transform_first(transforms.ToTensor()), 
                                       batch_size=batch_size, 
                                       shuffle=True, 
                                       num_workers=4)
    
    mnist_valid = gluon.data.vision.MNIST(train=False)
    valid_data = gluon.data.DataLoader(mnist_valid.transform_first(transforms.ToTensor()), 
                                       batch_size=batch_size, 
                                       num_workers=4)
    return train_data, train_data

def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(120, activation="relu"),
            nn.Dense(84, activation="relu"),
            nn.Dense(10)
        )
    return net

def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    acc = (output.argmax(axis=1) == label.astype('float32'))
    return acc.mean().asscalar()


def train(net, train_data, valid_data, epochs, batch_size, trainer, model_dir, ctx):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    
    for epoch in range(epochs):
        train_loss, train_acc, valid_acc = 0., 0., 0.
        tic = time()
        for data, label in train_data:
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label.as_in_context(ctx))
            loss.backward()

            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()
            train_acc += acc(output, label.as_in_context(ctx))
        
        for data, label in valid_data:
            output = net(data.as_in_context(ctx))
            valid_acc += acc(output, label.as_in_context(ctx))
       
  
        logging.info("Epoch[%d] Loss:%.3f Acc:%.3f|%.3f Perf: %.1f img/sec"%(
        epoch, train_loss/len(train_data),
        train_acc/len(train_data),
        valid_acc/len(valid_data),
        len(train_data._dataset)/(time()-tic)))
    
    net.save_parameters(os.path.join(model_dir, "net.params"))
    logging.info("Saved model params")
    logging.info("End of training")
    
if __name__ == '__main__':
    logging = get_logger(__name__)
    
    options = parse_args()
    
    ctx = mx.gpu() if options.num_gpus > 0 else mx.cpu()
    train_data, validation_data = get_data(options.batch_size)
    
    net = get_net()
    net.initialize(init=init.Xavier(), ctx=ctx)
    
    optimizer_params = {'learning_rate': options.learning_rate, 'wd': options.wd}

    if options.optimizer == 'sgd':
        optimizer_params['momentum'] = options.momentum

    trainer = gluon.Trainer(net.collect_params(), options.optimizer, optimizer_params)
    
    train(net, train_data, validation_data, options.epochs, options.batch_size, trainer, options.model_dir, ctx)
    
    
def model_fn(model_dir):
    net = get_net()
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    net.load_parameters(os.path.join(model_dir, "net.params"), ctx)
    return net
    

def transform_fn(net, data, input_content_type, output_content_type):
    data_dict = json.loads(data.decode())
    input_data = nd.array(data_dict['input'])
    
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    input_data = input_data.as_in_context(ctx)
    pred = net(input_data)
    
    response = json.dumps(pred.asnumpy().tolist())
    return response, output_content_type