import matplotlib.pyplot as plt
import numpy as np

with np.load('./result/cnn/kenel_num=6, kenel_size=5.npz') as data:
    x1 = data['arr_0']
    y1 = data['arr_1']

with np.load('./result/cnn/avg, kenel_num=6, kenel_size=5.npz') as data:
    x2 = data['arr_0']
    y2 = data['arr_1']

with np.load('./result/cnn/lp, kenel_num=6, kenel_size=5.npz') as data:
    x3 = data['arr_0']
    y3 = data['arr_1']

plt.plot(x1, y1, color='red', label='max_pooling')
plt.plot(x2, y2, color='green', label='avg_pooling')
plt.plot(x3, y3, color='blue', label='lp_pooling')
plt.title('pooling_type = max, avg or lp')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('./fig/cnn/pooling_type.png')
# plt.show()