import time
executionStart = time.time()
print("({:.3f}s) Importing stuff...".format(time.time() - executionStart))

import os
import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np

from solution.model import loadMyModel, saveMyModel, initFCN
from solution.data import load_datasets, INPUT_IMG_SIZE

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
np.random.seed(42)
random.seed(42)

EPOCH_COUNT = 12
IMG_WRITER_PERIOD = 4



TESTING_RATE = 1 # we execute testing sessions this many times during an epoch
LOGGING_PERIOD = 50 # there will be this many logs per train epoch

if TESTING_RATE < 1:
    raise Exception('Testing rate has to be positive')


writer = SummaryWriter('runs/hyper-deep')


print("({:.3f}s) Preparing data...".format(time.time() - executionStart))
train_size, valid_size, train_loader, valid_loaders, result_to_image = load_datasets()
print("({:.3f}s) Preparing model...".format(time.time() - executionStart))
net, device = initFCN()
# net, device = loadMyModel()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

def output_to_result(output):
    return torch.argmax(output, dim=1)

def score(output, target):
    out_render = output_to_result(output)

    grad = out_render - target
    points = torch.nonzero(grad).size(0)
    max_points = grad.flatten().size(0)

    return 100 * (max_points - points) / max_points

def write_comparison_image(title, data, output, target, iteration):
    computed = result_to_image(output_to_result(output))
    expected = result_to_image(target)
    grid = torchvision.utils.make_grid([data[0], computed[0], expected[0]])
    writer.add_image(title, grid, iteration)

def train():
    net.train()
    crit = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.94)

    batch_count = len(train_loader)

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        out = net(data)
        loss = crit(out, target)
        loss.backward()

        optimizer.step()


        # if batch_id > 0 and batch_id % int(batch_count / TESTING_RATE) == 0:
        #     test()

        iteration = (batch_count * epoch) + batch_id
        iter_score = score(out, target)

        lrate = optimizer.param_groups[0]['lr']

        print('(E{} {:.3f}s T)\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScore:  {:.3f}%'.format(
            epoch + 1, time.time() - executionStart, batch_id * len(data), train_size, # minor bug - last log for 100% has wrong image count
            100. * batch_id / batch_count, loss.item(), iter_score))

        write_comparison_image('train_images', data, out, target, iteration)

        writer.add_scalar('train_loss', loss.item(), iteration)
        writer.add_scalar('train_score', iter_score, iteration)

def test():
    net.eval()
    losses = []
    scores = []
    correct = 0
    crit = nn.CrossEntropyLoss()
    batch_count = len(valid_loaders[0]) * len(valid_loaders)
    lastdata = None
    with torch.no_grad():
        for loader_id, dataloader in enumerate(valid_loaders):
            for batch_id, (data, target) in enumerate(dataloader):
                batch_id += loader_id * len(valid_loaders[0])
                data, target = data.to(device), target.to(device)
                output = net(data)
                l = crit(output, target).item()
                s = score(output, target)
                losses.append(l)
                scores.append(s)

                iteration = epoch * batch_count + batch_id

                write_comparison_image('test_images', data, output, target, iteration)

                writer.add_scalar('test_loss', l, iteration)
                writer.add_scalar('test_score', s, iteration)

                print('(E{} {:.3f}s V)\t[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tScore:  {:.3f}%'.format(
                    epoch + 1, time.time() - executionStart, batch_id * len(data), valid_size, # minor bug - last log for 100% has wrong image count
                    100. * batch_id / batch_count, l, s))

                lastdata = data

    avg_score = statistics.mean(scores)
    avg_loss = statistics.mean(losses)

    writer.add_scalar('test_total_loss', avg_loss, epoch)
    writer.add_scalar('test_total_score', avg_score, epoch)
    
    writer.add_graph(net, lastdata, epoch)

    print('\nTest set: Average loss: {:.4f}, Score: ({:.2f}%)\n'.format(
        avg_loss, avg_score))
            
print("({:.3f}s) Beginning training session...".format(time.time() - executionStart))

for epoch in range(EPOCH_COUNT):
    train()
    test()
    saveMyModel(net, '-epoch-' + str(epoch))


saveMyModel(net, '')

writer.close()
