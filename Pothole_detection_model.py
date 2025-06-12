

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d rajdalsaniya/pothole-detection-dataset

import zipfile
zip_ref = zipfile.ZipFile('/content/pothole-detection-dataset.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

# pothole.yaml

  # root directory of your dataset
data_content = """

train: /content/train/images # path to training images
val: /content/valid/images      # path to validation images

nc: 1  # number of classes
names: ['pothole']  # class names
"""
with open('pothole_config.yaml', 'w') as f:
    f.write(data_content)



# Write to file
with open("custom_hyp.yaml", "w") as f:
    f.write(hyp_content)

!pip install ultralytics

# Commented out IPython magic to ensure Python compatibility.
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8l.pt')

# Define your hyperparameters in a Python dictionary
# This effectively replaces your 'custom_hyp.yaml' content
hyp_content = {
    'lr0': 0.01,           # initial learning rate
    'lrf': 0.01,           # final learning rate (lr0 * lrf)
    'momentum': 0.937,
    'weight_decay': 0.0005,

    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    'box': 0.05,
    'cls': 0.5,



      # 0 for CE loss (you can increase for focal loss)

    'hsv_h': 0.005,
    'hsv_s': 0.1,
    'hsv_v': 0.1,
    'degrees': 0.0,
    'translate': 0.05,
    'scale': 0.05,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.1,
    'fliplr': 0.2,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# Train with custom hyperparameters
model.train(
    data='pothole_config.yaml',  # your dataset config file
    epochs=50,
    batch=8,
    imgsz=640,
    name='yolov8l',            # experiment name
    **hyp_content               # Unpack the hyperparameters from the dictionary
)# Change directory back to the root before downloading
# %cd /content/

from google.colab import files
# Use the correct path relative to the root
files.download('runs/detect/yolov8l/weights/best.pt')  # or last.pt

from ultralytics import YOLO

# Load the two models (weights from different runs)

model2 = YOLO('/content/best.pt')

# Evaluate on validation or test dataset
# (make sure data.yaml path is correct and points to val/test images and labels)

results2 = model2.val(data='pothole_config.yaml')

# Access the detection metrics from the 'box' attribute
# results2.box contains the Metric object with the actual detection results
# .maps is mAP@[0.5:0.95], .map50 is mAP@0.5
print("Model from yolov10mbest.pt metrics:")
print(f"mAP@[0.5:0.95]: {results2.box.maps}")
print(f"mAP@0.5: {results2.box.map50}")

from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO("/content/best.pt")  # path to your trained model

# Path to your validation images
val_img_dir = "/content/valid/images"  # typically from data.yaml
img_paths = [os.path.join(val_img_dir, img) for img in os.listdir(val_img_dir) if img.endswith(('.jpg', '.png'))]

# Run predictions and visualize
for img_path in img_paths[:5]:  # limit to first 5 images
    results = model(img_path)  # predict
    res_plotted = results[0].plot()  # draw boxes on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(res_plotted)
    plt.title(f"Predictions on {os.path.basename(img_path)}")
    plt.axis('off')
    plt.show()

