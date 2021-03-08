# CNNOptionDectection
 A simple CNN network for detecting question options.

## Document Illustration
- load_img.py: [Preprocessing] for split the images under the directory input into the outputs directory.
```
python load_img.py --input_dir selection/
```
- CNN.py: for training the datasets selected and grouped manually from outputs directory.
```
python CNN.py --dataset_path datasets/
```
- predict.py for using the trained model
```
python predict.py --type single --input_dir inputs/
```
- callbacks.py for visualizing the training loss and validation loss. Do not need to import.
