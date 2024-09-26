import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load the model
model = fasterrcnn_resnet50_fpn(weights=None)


# Adjust for the number of classes (background + 1 class)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


# Load your trained weights
device = torch.device('cpu')
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn.pth', map_location=device))
model.eval()

# Convert the model to TorchScript
scripted_model = torch.jit.script(model)  # or torch.jit.trace if your model has fixed inputs

# Save the TorchScript model
scripted_model.save('fasterrcnn_resnet50_fpn.pt')
