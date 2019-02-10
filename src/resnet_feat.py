import argparse
from data_loader import DataLoader

from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50


ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", "-d", help="path to weapon data")
args = vars(ap.parse_args())

# Load & preprocess data
print("Loading data...")
image_data, image_labels, y_bin = DataLoader().get_image_data(args["data_dir"])
image_data = preprocess_input(image_data)
print("DONE")

print("RetNet50...")
features = ResNet50(weights="imagenet").predict(image_data)
print("DONE")

print(features.shape)

# Save features to file
data_file = args["data_dir"] + "features.data"
names_file = args["data_dir"] + "features.names"

with open(data_file, "w") as d_file:
    for feat, label in zip(features, image_labels):
        d_file.write(','.join(str(ff) for ff in feat.tolist()) + "," + str(label) + '\n')

with open(names_file, "w") as n_file:
    n_file.write("0,1.\n")
    for idx in range(1, features.shape[1] + 1):
        n_file.write("A{0}: continuous.\n".format(idx))
    
