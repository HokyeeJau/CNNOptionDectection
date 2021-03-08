import keras
import numpy as np
import pandas as pd
import os
import cv2
import re
from sklearn.utils import shuffle
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from callbacks import LossHistory
from keras.models import load_model
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="datasets/")
parser.add_argument("--white_start", type=int, default=200)
parser.add_argument("--white_end", type=int, default=255)

answer_dict = {
	"13-1": [0, 1, 0, 0],
	"13-2": [1, 0, 0, 0],
	"13-3": [1, 0, 0, 0],
	"13-4": [0, 0, 0, 0],
	"14-1": [1, 0, 0, 0],
	"14-2": [0, 0, 1, 0],
	"14-3": [1, 0, 0, 0],
	"14-4": [1, 0, 0, 0],
	"14-5": [0, 1, 0, 0],
	"14-6": [1, 0, 1, 0],
	"14-7": [1, 1, 1, 0],
	"14-8": [0, 0, 0, 0],
	"14-9": [0, 0, 0, 1],
	"15-1": [1, 0, 0, 0],
	"15-2": [1, 0, 0, 0],
	"15-3": [0, 0, 1, 0],
	"15-4": [0, 1, 0, 0],
	"15-5": [0, 1, 0, 0],
	"16-1": [0, 1, 0, 0],
	"16-2": [0, 0, 1, 0],
	"16-3": [1, 0, 0, 0],
	"16-4": [0, 0, 1, 0],
	"16-5": [1, 1, 0, 0],
	"16-6": [0, 0, 1, 0],
	"16-7": [1, 0, 0, 0],
	"16-8": [0, 0, 0, 0],
	"16-9": [1, 0, 1, 0],
	"16-10": [1, 0, 1, 0],
	"17-5": [0, 0, 0, 0],
	"19-1": [1, 0, 0, 0],
	"19-2": [1, 0, 0, 0],
	"19-3": [0, 0, 1, 0],
	"19-4": [0, 1, 0, 0],
	"19-5": [0, 1, 0, 0],
	"20-1": [1, 0, 0, 0],
	"20-2": [0, 0, 1, 0],
	"20-3": [1, 0, 0, 0],
	"20-4": [1, 0, 0, 0],
	"20-5": [0, 1, 0, 0],
	"20-6": [1, 0, 1, 0],
	"20-7": [1, 1, 1, 0],
	"20-8": [0, 0, 0, 0],
	"20-9": [0, 0, 0, 1],
	"200-1": [1, 0, 0, 0],
	"200-2": [1, 0, 0, 0],
	"200-3": [1, 0, 0, 0],
	"200-4": [1, 0, 0, 0],
	"200-5": [0, 1, 1, 0],
	"200-6": [1, 1, 1, 1],
	"200-7": [1, 1, 0, 1]
}

class GetDataset:
	def __init__(self, img_dir, white_start, white_end width=300, height=300, transform=True):
		self.img_dir = img_dir
		self.transform = transform
		self.dataset = []
		self.labels = []

		self.white_start = white_start
		self.white_end = white_end
		self.width = width
		self.height = height

	def _extract_prefix(self, img_name):
		pattern = re.compile(r"([A-Za-z0-9_-]+)\..+")
		return pattern.findall(img_name)[0]

	def _import(self):
		self.img_list = os.listdir(self.img_dir)
		for img_name in self.img_list:
			path = os.path.join(self.img_dir, img_name)
			img = image.load_img(path, target_size=(self.width, self.height))
			img = image.img_to_array(img)
			img = preprocess_input(img)
			self.labels.append(self._extract_prefix(img_name))
			self.dataset.append(img)
		self.dataset = np.array(self.dataset)
		if self.transform:
			self.labels = self._transform_labels()
		else:
			self.labels = np.array(self.labels)

	def _transform_labels(self):
		labels = []
		for l in self.labels:
			labels.append(answer_dict[l])
		return np.array(labels)

	def _reshape_dataset(self):
		self.dataset = np.expand_dims(self.dataset, axis=-1)

	def get_dataset(self):
		self._import()
		# self._reshape_dataset()
		# return shuffle(self.dataset, self.labels)
		return self.dataset, self.labels

class TrainerConfig:
	def __init__(self, block1_filters = 128,
	          block1_strides = (2, 2),
	          block1_pool_size = (2, 2),
	          block1_pool_strides = 2,
	          block2_filters = 128,
	          block2_strides = (2, 2),
	          block2_pool_size = (2, 2),
	          block2_pool_strides = 2,
	          dense1_units = 8,
	          dropout = 0.2,
	          dense2_units = 4,
	          epoch = 200):
		self.block1_filters = block1_filters
		self.block1_strides = block1_strides
		self.block1_pool_size = block1_pool_size
		self.block1_pool_strides = block1_pool_strides

		self.block2_filters = block2_filters
		self.block2_strides = block2_strides
		self.block2_pool_size = block2_pool_size
		self.block2_pool_strides = block2_pool_strides

		self.dense1_units = dense1_units
		self.dropout = dropout
		self.dense2_units = dense2_units
		self.epoch = epoch

	def _print(self):
		print('\n'.join(['%s:%s'%item for item in self.__dict__.items()]))

def ConvNN(c, input_shape):
	model = Sequential()
	model.add(Conv2D(filters=c.block1_filters, kernel_size=3, strides=c.block1_strides, input_shape=input_shape))
	model.add(Conv2D(filters=c.block1_filters, kernel_size=3, strides=c.block1_strides))
	model.add(MaxPool2D(pool_size=c.block1_pool_size, strides=c.block1_pool_strides))
	model.add(BatchNormalization())

	model.add(Conv2D(filters=c.block2_filters, kernel_size=3, strides=c.block2_strides))
	model.add(Conv2D(filters=c.block2_filters, kernel_size=3, strides=c.block2_strides))
	# model.add(MaxPool2D(pool_size=(2, 2), strides=2))
	model.add(BatchNormalization())

	# model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='tanh'))
	# model.add(BatchNormalization())
	model.add(Flatten())

	model.add(Dense(units=c.dense1_units, activation='tanh'))
	model.add(Dropout(rate=c.dropout))
	model.add(BatchNormalization())
	model.add(Dense(units=c.dense2_units, activation='softmax'))

	return model


class Trainer:
	def __init__(self, features, labels, config):
		self.features = features
		self.labels = labels
		self.config = config
		print(f'Shape of Images: {self.features.shape}')
		print(f'Shape of Labels: {self.labels.shape}')

	def run(self):
		self._modeling()
		self._training()

	def _modeling(self):
		self.model = ConvNN(self.config, self.features.shape[1:])

	def _training(self):
		self.model.compile(optimizer='adam', loss='categorical_crossentropy')
		self.history = LossHistory()
		self.trained = self.model.fit(self.features[:40],
		                      self.labels[:40],
		                      epochs=self.config.epoch,
                          # batch_size=32,
		                      validation_data=(self.features[30:], self.labels[30:]),
		                      verbose=1,
		                      shuffle=True,
		                      callbacks=[self.history])

	def _epoch_loss(self):
		return self.history.loss_plot('epoch')

	def _batch_loss(self):
		return self.history.loss_plot('batch')

	def _summary(self):
		return self.model.summary()

	def _test(self, start, end):
		y_pred = self.model.predict(self.features[start:end])
		y_true = self.labels[start:end]
		for i in range(len(y_pred)):
			print(y_true[i], y_pred[i])
		# print((y_pred > 0.1).astype(int))
		# print((y_pred > 0.5).astype(int))



if __name__ == '__main__':
	Dataset = GetDataset(parser.dataset_path, parser.white_start, parser.white_end)
	train_X, trian_y = Dataset.get_dataset()
	config = TrainerConfig(block1_filters=128,
	                       block1_strides=(2, 2),
	                       block1_pool_size=(2, 2),
	                       block1_pool_strides=1,
	                       block2_filters=128,
	                       block2_strides=(2, 2),
	                       block2_pool_size=(2, 2),
	                       block2_pool_strides=1,
	                       dense1_units=8,
	                       dropout=0.2,
	                       dense2_units=4)
	trainer = Trainer(train_X, trian_y, config)
	trainer.run()
	trainer._summary()
	trainer._batch_loss()
	trainer._epoch_loss()
	trainer._test(0, -1)
