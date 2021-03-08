from keras.models import load_model
from CNN import GetDataset
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default="single", help="single or multiple")
parser.add_argument("--white_end", type=int, default=255)
parser.add_argument("--white_start", type=int, default=200)
parser.add_argument("--input_dir", type=str, default="inputs/")

def predict(dir_path):
	Dataset = GetDataset("inputs/", parser.white_start, parser.white_end transform=False)
	test_X, test_y = Dataset.get_dataset()
	if parser.type=="single":
		model = load_model("single_choice_detector.h5")
	else:
		model = load_model("multi_choice_detector.h5")
	x = model.predict(test_X)
	for i in range(len(x)):
	  print(test_y[i], x[i])
