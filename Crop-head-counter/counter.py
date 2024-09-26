import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os, sys

device = torch.device("cpu")
BASE_PATH="./"
PTH_MODEL_PATH="trained-models/fasterrcnn_resnet50_fpn.pth"
threshold = 0.5

# Load the model with the correct number of classes
num_classes = 2  
model = fasterrcnn_resnet50_fpn(weights=None)

# Replace the box predictor with a new one
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)




def loadModel():
    # Load the model state dict
    model.load_state_dict(torch.load(BASE_PATH+'/'+PTH_MODEL_PATH, map_location=device,weights_only=True))
    model.to(device)
    model.eval()


# Function to perform inference
def predict(image_path):
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)


    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']
    box_count = (scores > threshold).sum().item()

    return outputs, image, box_count, threshold




# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, labels, scores,threshold):
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            draw.rectangle(box.tolist(), outline="red", width=3)
            draw.text((box[0], box[1]), f'{label.item()} {score:.2f}', fill="red")
    return image




if __name__ == "__main__":
    image_path = sys.argv[1]
    if len(sys.argv)>2:
        BASE_PATH = sys.argv[2]

    loadModel()
    results, image, box_count, threshold = predict(image_path)

    boxes = results[0]['boxes']
    labels = results[0]['labels']
    scores = results[0]['scores']

    output_image = draw_boxes(image, boxes, labels, scores,threshold)

    output_image.save(BASE_PATH+"/output/"+os.path.basename(image_path))

    print(box_count)