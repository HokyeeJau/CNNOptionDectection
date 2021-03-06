from keras.callbacks import Callback
import matplotlib.pyplot as plt
import time

class LossHistory(Callback):

	def on_train_begin(self, logs={}):
		self.losses = {'batch': [], 'epoch': []}
		self.accuracy = {'batch': [], 'epoch': []}
		self.val_loss = {'batch': [], 'epoch': []}
		self.val_acc = {'batch': [], 'epoch': []}

	def on_batch_end(self, batch, logs={}):
		self.losses['batch'].append(logs.get('loss'))
		self.accuracy['batch'].append(logs.get('acc'))
		self.val_loss['batch'].append(logs.get('val_loss'))
		self.val_acc['batch'].append(logs.get('val_acc'))

	def on_epoch_end(self, batch, logs={}):
		self.losses['epoch'].append(logs.get('loss'))
		self.accuracy['epoch'].append(logs.get('acc'))
		self.val_loss['epoch'].append(logs.get('val_loss'))
		self.val_acc['epoch'].append(logs.get('val_acc'))

	def loss_plot(self, loss_type):
		'''
		loss_type：指的是 'epoch'或者是'batch'，分别表示是一个batch之后记录还是一个epoch之后记录
		'''
		iters = range(len(self.losses[loss_type]))
		plt.figure()
		# acc
		plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
		# loss
		plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
		if loss_type == 'epoch':
			# val_acc
			plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
			# val_loss
			plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
		plt.grid(True)
		plt.xlabel(loss_type)
		plt.ylabel('acc-loss')
		plt.legend(loc="upper right")
		plt.show()
		plt.savefig(f"{time.time()}.png")