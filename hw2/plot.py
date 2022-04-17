import matplotlib.pyplot as plt
import numpy as np

with np.load('./result/resnet/lr=1e-2, batch_size=64.npz') as data:
    x1 = data['arr_0']
    y1 = data['arr_1']

with np.load('./result/resnet/lr=1e-2, batch_size=128.npz') as data:
    x2 = data['arr_0']
    y2 = data['arr_1']

with np.load('./result/resnet/lr_alter, batch_size=128.npz') as data:
    x3 = data['arr_0']
    y3 = data['arr_1']

plt.plot(x1, y1, color='red', label='lr=1e-2, batch_size=64')
plt.plot(x2, y2, color='green', label='lr=1e-2, batch_size=128')
plt.plot(x3, y3, color='blue', label='lr_alter, batch_size=128')
plt.title('ResNet')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('./fig/cnn/resnet.png')
# plt.show()