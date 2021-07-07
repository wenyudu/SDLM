import argparse
import os
import time
import math
import random
import hashlib
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

import data
import model
from utils import rankloss, get_batch, repackage_hidden
from splitcross import SplitCrossEntropyLoss

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/syntactic_penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.5,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.125,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.45,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
# randomhash = ''.join(str(time.time()).split('.'))
# parser.add_argument('--save', type=str, default=randomhash + '.pt',
#                     help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
# parser.add_argument('--resume', type=str, default='',
#                     help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',choices=['sgd','adam'],
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--finetuning', type=int, default=650,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
parser.add_argument('--device', type=int, default=0, help='select GPU')
parser.add_argument('--l4d',type=int,default=2,choices=[-1,0,1,2,],help='layer for distance')
parser.add_argument('--margin', type=float, default=1.0,
                    help='margin at rank loss')
parser.add_argument('--wds', type=str, default='middle',choices=['no','before','middle','after'],
                    help='different ways to use weighted distance signal')
parser.add_argument('--un', action='store_true',
                    help='unsupervised settings')
parser.add_argument('--sratio', type=float, default=1.0,
                    help='supervised signal ratio')
parser.add_argument('--dratio', type=float, default=1.0,
                    help='data size ratio')
parser.add_argument('--alpha1', type=float, default=1.0)
parser.add_argument('--alpha2', type=float, default=1.0)
args = parser.parse_args()
args.tied = True

args.batch_size_tune = args.batch_size_init = args.batch_size

if not os.path.isdir('params/'):
    os.mkdir('params')
save_string = 'params/' + hashlib.md5(str(args).encode()).hexdigest() + '.pt'
print("Params saving to: " + save_string)

assert 0.0 <= args.margin and args.margin <= 1.0
assert 0.0 <= args.sratio and args.sratio <= 1.0

args.penn_only = False
if args.un:
    args.penn_only = True
    args.wds = "no"

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([epoch, model, criterion, optimizer], f)


def model_load(fn):
    global epoch, model, criterion, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        epoch, model, criterion, optimizer = torch.load(f)

fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if args.philly:
    fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
if os.path.exists(fn) and args.data != 'data/syntactic_penn/':
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.syntactic_penn(args, args.batch_size_init,args.dratio)
    torch.save(corpus, fn)

train_data = corpus[0]
train_dist = corpus[1]
val_data = corpus[2]
test_data = corpus[3]
args.vocab_size = len(corpus[4])

print("done loading, vocabulary size: {}".format(args.vocab_size))

eval_batch_size=80
test_batch_size=1

###############################################################################
# Build the model
###############################################################################

criterion = None

ntokens = args.vocab_size
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                       args.chunk_size, args.nlayers, args.wds,args.dropout,
                       args.dropouth, args.dropouti, args.dropoute,
                       args.wdrop, args.tied, args.l4d)
###
start_epoch = 0
if os.path.exists(save_string):
    print('Resuming model ...')
    model_load(save_string)
    start_epoch = epoch

    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = \
        args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnn.cells:
            rnn.hh.dropout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################
@torch.no_grad()
def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = args.vocab_size
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight,
                                            model.decoder.bias,
                                            output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(train_batch_size):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = total_sdloss = 0
    start_time = time.time()
    ntokens = args.vocab_size
    hidden = model.init_hidden(train_batch_size)
    batch, i = 0, 0
    train_data_full_size = train_data.size(0) - 1 - 1
    while i < train_data_full_size:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence
        # length resulting in OOM seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        dist, _ = get_batch(train_dist, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was
        # previously produced. If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        # output, hidden = model(data, hidden, return_h=False)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )

        forget_distance = model.distance[1]

        if args.l4d >= 0:
            single_layer_distance = forget_distance[args.l4d]
            distance_loss = rankloss(single_layer_distance, dist, margin=args.margin)
        else:
            distance_loss = rankloss(forget_distance[0], dist, margin=args.margin) \
                            + rankloss(forget_distance[2], dist, margin=args.margin)

        if i < train_data_full_size * args.sratio:
            sd_loss = args.alpha1 * loss + args.alpha2 * distance_loss
        else:
            sd_loss = loss

        if args.penn_only:
            loss.backward()
        else:
            sd_loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        total_sdloss += sd_loss.data

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            cur_sdloss = total_sdloss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f} | dist {:2.5f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                                  elapsed * 1000 / args.log_interval, cur_loss,
                                  math.exp(cur_loss), cur_loss /
                                  math.log(2),cur_sdloss))
            total_loss = total_sdloss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the
    # model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999),
                                     eps=1e-9, weight_decay=args.wdecay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5,
                                                   patience=2, threshold=0)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if epoch <= args.finetuning:
            train_batch_size = args.batch_size_init
        else:
            train_batch_size = args.batch_size_tune
        epoch_start_time = time.time()
        train(train_batch_size)
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2,
                math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(save_string)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

            if epoch == args.finetuning:
                print('Switching to finetuning')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr,
                                             t0=0, lambd=0.,
                                             weight_decay=args.wdecay)
                best_val_loss = []

            if (epoch > args.finetuning and
                    len(best_val_loss) > args.nonmono and
                    val_loss2 > min(best_val_loss[:-args.nonmono])):
                print('Done!')
                import sys

                sys.exit(1)

        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss,
                math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(save_string)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'adam':
                scheduler.step(val_loss)

            if (args.optimizer == 'sgd' and
                    't0' not in optimizer.param_groups[0] and
                    (len(best_val_loss) > args.nonmono and
                        val_loss > min(best_val_loss[:-args.nonmono]))):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(
                    model.parameters(), lr=args.lr, t0=0, lambd=0.,
                    weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(save_string, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

        print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(save_string)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
