import time

import numpy as np
import torch
import torch.nn as nn
import data_n
import model
from arguments import args
from loss_and_opt import loss_fn, get_optimizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 引入模型
fcn = model.get_model()
fcn = nn.DataParallel(fcn).to(device)

# 设置优化器
optimizer = get_optimizer(fcn)

# 设置格式
header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}' * 2 + '\u2502{:6.2f}'


# 开始训练
def train():
    print("------------------Start training------------------")
    best_loss = 10
    for epoch in range(1, args.epochs + 1):
        losses = []
        start_time = time.time()
        fcn.train()
        for image, target in tqdm(data_n.loader):
            image, target = image.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = fcn(image)['out']
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(loss.item())

        vloss = model.validation(fcn, data_n.vloader, loss_fn)
        print(header)
        print(raw_line.format(epoch, np.array(losses).mean(), vloss,
                              (time.time() - start_time) / 60 ** 1))

        if vloss < best_loss:
            best_loss = vloss
            torch.save(fcn.module.state_dict(), args.resume + 'model_best.pth')


if __name__ == '__main__':
    train()
    print('Done!')
