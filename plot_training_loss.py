import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load('training_loss.npy')

x = np.arange(0,train_loss.shape[0])

moving_avg_train_loss = np.zeros(train_loss.shape[0])
sum = train_loss[0]
moving_avg_train_loss[0] = train_loss[0]
for i in range(1,train_loss.shape[0]):
    moving_avg_train_loss[i] = sum/(i + 1)
    sum += train_loss[i]

plt.figure(figsize=(10, 10))
plt.plot(x,train_loss)
plt.plot(x,moving_avg_train_loss)
plt.xlabel('iterations')
plt.ylabel('train_loss')
plt.title('Training Loss')
plt.show()