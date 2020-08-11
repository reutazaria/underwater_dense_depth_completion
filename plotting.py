import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '../results/mode=dense.data=cave.input=rgbd.resnet18.epochs100.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained' \
           '=True.jitter=0.1.time=2020-08-11@16-03/train.csv'
csv_data = pd.read_csv(filename)
epochs = csv_data['epoch'].values
train_loss = csv_data['loss'].values
depth_loss = csv_data['depth_loss'].values
smooth_loss = csv_data['smooth_loss'].values
photometric_loss = csv_data['photometric_loss'].values
train_error = csv_data['rmse'].values
epoch_num = len(epochs)

print(
    'Best train error is: {:.3f}, in epoch {:d}/{:d}'.format(min(train_error),
                                                             min(np.where(train_error == min(train_error))[0]) + 1,
                                                             epoch_num + 1))

# plotting
epochs = range(epoch_num) + np.ones(epoch_num)
fig0 = plt.figure(1)
# plt.figure(figsize=(20, 10))
plt.plot(epochs, train_loss, label='Train Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TrainLoss')
plt.grid(True)
# fig0.savefig('hw1_Loss.png')

fig1 = plt.figure(2)
plt.plot(epochs, train_error, label='Train Error')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Error [%]')
plt.title('Train Error')
plt.grid(True)
fig1.savefig('hw1_Error.png')
