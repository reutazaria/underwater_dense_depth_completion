import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_dir = '../results/mode=dense.data=cave.input=rgbd.resnet18.epochs100.criterion=l2.lr=0.0001.bs=2.wd=0' \
            '.pretrained=True.jitter=0.1.time=2020-08-11@16-03'
train_filename = '../results/mode=dense.data=cave.input=rgbd.resnet18.epochs100.criterion=l2.lr=0.0001.bs=2.wd=0' \
                 '.pretrained=True.jitter=0.1.time=2020-08-11@16-03/train.csv'
train_data = pd.read_csv(train_filename)
epochs = train_data['epoch'].values
train_loss = train_data['loss'].values
depth_loss = train_data['depth_loss'].values
smooth_loss = train_data['smooth_loss'].values
photometric_loss = train_data['photometric_loss'].values
train_error = train_data['rmse'].values
epoch_num = len(epochs)

print(
    'Best train loss is: {:.3f}, in epoch {:d}/{:d}'.format(min(train_loss),
                                                            min(np.where(train_loss == min(train_loss))[0]) + 1,
                                                            epoch_num))


val_filename = '../results/mode=dense.data=cave.input=rgbd.resnet18.epochs100.criterion=l2.lr=0.0001.bs=2.wd=0' \
               '.pretrained=True.jitter=0.1.time=2020-08-11@16-03/val.csv'
val_data = pd.read_csv(val_filename)
val_loss = val_data['loss'].values
# depth_loss = val_data['depth_loss'].values
# smooth_loss = val_data['smooth_loss'].values
# photometric_loss = val_data['photometric_loss'].values
val_error = val_data['rmse'].values

print(
    'Best validation error is: {:.3f}, in epoch {:d}/{:d}'.format(min(val_error),
                                                                  min(np.where(val_error == min(val_error))[0]) + 1,
                                                                  epoch_num ))

# plotting
epochs = range(epoch_num) + np.ones(epoch_num)

fig0 = plt.figure(1)
plt.subplot(121)
plt.plot(epochs, train_loss, label='Train Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss')
plt.grid(True)
# fig0.savefig(model_dir + '/' + 'train_loss.png')
#
# fig1 = plt.figure(2)
plt.subplot(122)
plt.plot(epochs, val_error, label='Validation Error')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('RMSE [mm]')
plt.title('Validation Error')
plt.grid(True)
fig1.savefig(model_dir + '/' + 'val_error.png')
