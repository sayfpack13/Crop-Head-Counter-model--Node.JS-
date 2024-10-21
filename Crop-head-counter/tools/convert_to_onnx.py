import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
device = torch.device('cpu')
model = fasterrcnn_resnet50_fpn(weights=None)

# Adjust for the number of classes (background + 1 class)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load model weights safely with weights_only=True
model.load_state_dict(torch.load('../trained-models/fasterrcnn_resnet50_fpn.pth', map_location=device))
model.to(device)
model.eval()

# Create dummy input for exporting the model
dummy_input = torch.randn(1, 3, IMAGE_WIDTH, IMAGE_HEIGHT).to(device)

# Export the model with dynamic axes for varying input sizes and number of boxes
torch.onnx.export(
    model, 
    dummy_input, 
    "../trained-models/fasterrcnn_resnet50_fpn.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["boxes", "labels", "scores"],  # Adjust according to actual ONNX outputs
    dynamic_axes={
        'input': {0: 'batch_size'},   # Dynamic batch size
        'boxes': {0: 'num_boxes'},    # Dynamic for number of detected boxes
        'labels': {0: 'num_boxes'},
        'scores': {0: 'num_boxes'}
    }
)
