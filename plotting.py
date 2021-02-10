import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

model_dir = '../results/mode=dense.data=nachsholim.input=rgbd.resnet18.epochs30.criterion=l2.lr=0.0001.bs=2.wd=0.pretrained=True.jitter=0.1.time=2021-02-09@15-42'
train_filename = os.path.join(model_dir, 'train.csv')
train_data = pd.read_csv(train_filename)
epochs = train_data['epoch'].values
train_loss = train_data['loss'].values
depth_loss = train_data['depth_loss'].values
smooth_loss = train_data['smooth_loss'].values
photometric_loss = train_data['photometric_loss'].values
train_error = train_data['rmse'].values
epoch_num = len(epochs)

val_filename = os.path.join(model_dir, 'val.csv')
val_data = pd.read_csv(val_filename)
val_error = val_data['rmse'].values
val_pearson = val_data['pearson'].values

fig0 = plt.figure(1)
data_set = model_dir.split('data=')[1].split('.')[0]
mode = model_dir.split('mode=')[1].split('.')[0]
model_input = model_dir.split('input=')[1].split('.')[0]
plt.suptitle(data_set + ' - ' + mode + ' losses, input=' + model_input)
epochs = range(epoch_num) + np.ones(epoch_num)

epoch_num = min(epoch_num, 35)
epochs = range(1, epoch_num+1)

plt.subplot(131)
plt.plot(epochs, depth_loss[0:epoch_num], label='Depth Loss')
min_ind = np.argmin(depth_loss[0:epoch_num])
plt.plot(epochs[min_ind], depth_loss[min_ind], 'xk')
plt.annotate("({:d},{:.3f})".format(epochs[min_ind], depth_loss[min_ind]),
            (epochs[min_ind], depth_loss[min_ind]), ha="center", bbox=dict(facecolor='grey', alpha=0.5))
if '+' in mode:
    plt.plot(epochs, train_loss[0:epoch_num], label='Train Loss')
    min_ind = np.argmin(train_loss[0:epoch_num])
    plt.plot(epochs[min_ind], train_loss[min_ind], 'xk')
    plt.annotate("({:d},{:.3f})".format(epochs[min_ind], train_loss[min_ind]),
                 (epochs[min_ind], train_loss[min_ind]), ha="center", bbox=dict(facecolor='grey', alpha=0.5))
    plt.plot(epochs, smooth_loss[0:epoch_num], label='Smooth Loss')
    plt.plot(epochs, photometric_loss[0:epoch_num], label='Correlation Loss')
    min_ind = np.argmin(photometric_loss[0:epoch_num])
    plt.plot(epochs[min_ind], photometric_loss[min_ind], 'xk')
    plt.annotate("({:d},{:.3f})".format(epochs[min_ind], photometric_loss[min_ind]),
                 (epochs[min_ind], photometric_loss[min_ind]), ha="center", bbox=dict(facecolor='grey', alpha=0.5))
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Losses')
plt.title('Train Losses')
plt.grid(True)
print('Best train loss is: {:.3f}, in epoch {:d}/{:d}'.format(min(train_loss),
                                                              min(np.where(train_loss == min(train_loss))[0]) + 1,
                                                              epoch_num))

plt.subplot(132)
val_error_trim = val_error[0:epoch_num]
plt.plot(epochs, val_error_trim, label='RMSE')
min_ind = np.argmin(val_error_trim)
plt.plot(epochs[min_ind], val_error_trim[min_ind], 'xk')
plt.annotate("({:d},{:.3f})".format(epochs[min_ind], val_error_trim[min_ind]),
             (epochs[min_ind], val_error_trim[min_ind]), ha="center", bbox=dict(facecolor='grey', alpha=0.5))
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('RMSE [mm]')
plt.title('Validation RMSE')
plt.grid(True)
print('Best validation error is: {:.3f}, in epoch {:d}/{:d}'.format(val_error_trim[min_ind],
                                                                    epochs[min_ind],
                                                                    epoch_num))

plt.subplot(133)
val_pearson_trim = val_pearson[0:epoch_num]
plt.plot(epochs, val_pearson_trim, label='Pearson')
max_ind = np.argmax(val_pearson_trim)
plt.plot(epochs[max_ind], val_pearson_trim[max_ind], 'xk')
plt.annotate("({:d},{:.3f})".format(epochs[max_ind], val_pearson_trim[max_ind]),
             (epochs[max_ind], val_pearson_trim[max_ind]), ha="center", va="bottom", bbox=dict(facecolor='grey', alpha=0.5))
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Pearson')
plt.title(r'Validation $\rho$')
plt.grid(True)

fig0.set_size_inches(20, 5)
plt.show()

fig0.savefig(model_dir + '/' + 'train_loss.png')


