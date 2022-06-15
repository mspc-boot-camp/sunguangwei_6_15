import time
import torch
import os
import random
from torch.autograd import Variable
from bulid_network import cnn
import numpy as np
import torch.nn as nn
from torch import optim
from data_preprocess import load_data
def setup_seed(seed):#随机数种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)
def train():
    start =time.time()
    train_loader, test_loader = load_data()
    print('train...')
    epoch_num = 30
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_loader, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    end = time.time()
    print('Training time: %s Seconds' % (end - start))
    torch.save(model.state_dict(), 'F:/PythonSave/dog_cat_classification/weights/weight_dog_cat.pt')
if __name__ == '__main__':
    train()