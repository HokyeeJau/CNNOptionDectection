import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="selection/", type=str)

class Image:
	"""
	At last, all the titles can be found in the following directory:
	outputs/
		image_name_prefix/
			gray/
			titles/
	"""
	def __init__(self, dir_path, img_name, img_prefix):
		self.img_dir = dir_path
		self.img_name = img_name
		self.img_prefix = img_prefix
		self.img_path = os.path.join(self.img_dir, self.img_name)

		self.save_dir = "outputs/"
		self.img_save_dir = os.path.join(self.save_dir, self.img_prefix)
		self.gray_dir = os.path.join(self.img_save_dir, "gray/")
		self.title_dir = os.path.join(self.img_save_dir, "titles/")

		self._build_dir(self.save_dir)
		self._build_dir(self.img_save_dir)
		self._build_dir(self.gray_dir,  clear=True)
		self._build_dir(self.title_dir, clear=True)

		# Prepare for cutting and classifying images
		self.white_start = 200
		self.white_end = 255

		# Cut image
		self.titles = []

	def process(self):
		self._check_img()
		self._transform_img_into_gray()

		self.df = pd.DataFrame(self.gray)

		self._cut_margin()
		self._cut_into_titles()
		self._save_images()

	def _save_images(self):
		"""
		Cut the residual white blank and save all the titles
		"""
		idx = 1
		for i in range(len(self.titles)):
			if np.sum((255-self.titles[i]).to_numpy()) >= 0:
				cv2.imwrite(self.title_dir+self.img_prefix+"-"+str(idx)+".jpg",
				            self._filter_pixels(self.titles[i], if_pure=True).to_numpy(),
				            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
				idx += 1

	def _transform_img_into_gray(self):
		"""
		Transform image into gray scale
		"""
		img_obj = cv2.imread(self.img_path)
		gray = cv2.cvtColor(img_obj, cv2.COLOR_BGR2GRAY)

		ret, img_thres = cv2.threshold(gray, self.white_start, self.white_end, cv2.THRESH_BINARY)
		self.gray = img_thres
		cv2.imwrite(self.gray_dir+self.img_prefix+".png", self.gray)

	def _cut_into_titles(self):
		"""
		Extract titles by hard cut.
		"""
		df_rev = 255 - self.df
		non_empty_idx = list(self._non_empty_set(df_rev, 1, 0))

		# Set down the threshold of the short length block for reducing the noisy dots
		threshold = int(self.df.shape[0] * 0.01)
		row_steps = self.df.shape[1] // 4

		temp_idx = 0
		for idx in range(1, len(non_empty_idx)):
			if (non_empty_idx[idx - 1] + 1 != non_empty_idx[idx] or idx == len(non_empty_idx) - 1) \
					and non_empty_idx[idx - 1] - non_empty_idx[temp_idx] > threshold:

				block = self.df.iloc[non_empty_idx[temp_idx]:non_empty_idx[idx - 1] + 1, :]
				temp_idx = idx
				# cv2.imshow(str(idx), block.to_numpy())
				self.titles += [block.iloc[:, :row_steps],
				                   block.iloc[:, row_steps:row_steps*2],
				                   block.iloc[:, row_steps*2:row_steps*3],
				                   block.iloc[:, row_steps*3:]]
				#                    block.iloc[:, row_steps*4:]]

	def _cut_margin(self):
		"""
		Cut Image into blocks
		1. Cut the left and the right margins
		2. Cut the white blank between the blocks

		For easily computing the pixels, we here reverse the 0 and 255 standings.
		"""
		# Reverse the data frame
		# filter_pixels drop the white surroundings
		self.df = self._filter_pixels(self.df)

		# Using proportion drop the black surroundings
		temp = self.df.copy()
		ratio = 0.001
		while np.sum(255-temp.iloc[0, :]) != 0:
			shape = temp.shape
			row_steps = max(1, int(ratio*shape[0]))
			col_steps = max(1, int(ratio*shape[1]))
			temp = temp.iloc[row_steps:shape[0]-row_steps, col_steps:shape[1]-col_steps]
			ratio += 0.001
		self.df = temp

		self.df = self._filter_pixels(self.df)

	def _filter_pixels(self, df_temp, if_pure=False): # df outside must be 255
		"""
		For cutting the white surroundings
		"""
		df_rev = 255 - df_temp
		df = df_temp.copy()

		column_threshold =  [0, df.shape[0] * 0.1 * 255] [if_pure == False]
		row_threshold =  [0, df.shape[1] * 0.1 * 255] [if_pure == False]

		# Get non-zero pixels index horizontally and vertically
		non_empty_column_idx = self._non_empty_set(df_rev, 0, column_threshold)
		non_empty_row_idx = self._non_empty_set(df_rev, 1, row_threshold)

		# To prevent the index set being empty
		if non_empty_row_idx.shape[0] == 0:
			non_empty_row_idx = np.append(non_empty_row_idx, [0, df.shape[0]])
		if non_empty_column_idx.shape[0] == 0:
			non_empty_column_idx = np.append(non_empty_column_idx, [0, df.shape[1]])

		# print(non_empty_row_idx.shape, non_empty_column_idx.shape)
		df = df.iloc[non_empty_row_idx[0]:non_empty_row_idx[-1], non_empty_column_idx[0]:non_empty_column_idx[-1]]

		# df = df.reset_index()
		df.columns = list(range(df.shape[1]))
		return df

	def _non_empty_set(self, df, axis, threshold):
		non_empty = np.where(np.array(df.sum(axis=axis) <= threshold) == False)
		non_empty = np.array(non_empty[0])
		return non_empty

	def _check_img(self):
		print(self.img_path)
		assert os.path.exists(self.img_path), "Image does not exist."

	def _build_dir(self, path, clear=False):
		if not Path(path).is_dir():
			os.mkdir(path)
		if clear:
			for root, dirs, files in os.walk(path, topdown=False):
				for name in files:
					os.remove(os.path.join(root, name))
				for name in dirs:
					os.rmdir(os.path.join(root, name))

	@property
	def _image_path(self):
		return self.img_path

	@property
	def _gray(self):
		return self.gray

	@property
	def _title_count(self):
		return len(self.titles)

	@property
	def _titles(self):
		return self.titles

class ImageProcessing:
	def __init__(self, img_dir):
		self.img_dir = 	img_dir
		assert Path(self.img_dir).is_dir(), "Directory does not exist."
		self._iterate_image()

	def _iterate_image(self):
		self.img_list = os.listdir(self.img_dir)
		self.img_obj = []

		for img in self.img_list:
			obj = Image(self.img_dir, img, self._img_prefix_extractor(img))
			obj.process()
			self.img_obj.append(obj)

	def _img_prefix_extractor(self, img_name):
		pattern = re.compile(r"([A-Za-z0-9_]+)\..+")
		return pattern.findall(img_name)[0]

	def get(self):
		return self.img_obj

if __name__ == "__main__":
	params = parser.parse_known_args()[0]
	dir_path = params.input_dir
	imgProc = ImageProcessing(dir_path)

