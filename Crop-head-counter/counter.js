import path from "path"
import fs from "fs"
import { createCanvas,loadImage } from "canvas"
import ort from "onnxruntime-node"

const ONNX_MODEL_PATH="trained-models/fasterrcnn_resnet50_fpn.onnx"
const threshold = 0.5
const boxLineThickness=5


async function loadImageAndPreprocess(imagePath) {
    const image = await loadImage(imagePath)

    const canvas = createCanvas(1024, 1024)
    const ctx = canvas.getContext('2d')

    ctx.drawImage(image, 0, 0, 1024, 1024)
    const imageData = ctx.getImageData(0, 0, 1024, 1024)
    const data = imageData.data

    const inputTensor = new Float32Array(1 * 3 * 1024 * 1024)

    for (let y = 0; y < 1024; y++) {
        for (let x = 0; x < 1024; x++) {
            const index = (y * 1024 + x) * 4
            inputTensor[0 * 1024 * 1024 + y * 1024 + x] = data[index] / 255 // R
            inputTensor[1 * 1024 * 1024 + y * 1024 + x] = data[index + 1] / 255 // G
            inputTensor[2 * 1024 * 1024 + y * 1024 + x] = data[index + 2] / 255 // B
        }
    }

    return { inputTensor, originalImage: image }
}

export async function runCropHeadCountModel(BASE_PATH, imagePath) {
    let crop_heads = 0;

    const modelPath = path.join(BASE_PATH, ONNX_MODEL_PATH);
    const session = await ort.InferenceSession.create(modelPath);
    const { inputTensor, originalImage } = await loadImageAndPreprocess(imagePath);

    const feeds = { input: new ort.Tensor('float32', inputTensor, [1, 3, 1024, 1024]) };
    const output = await session.run(feeds);

    const boxes = output.boxes.data; // Bounding box coordinates
    const scores = output.scores.data; // Confidence scores
    const labels = output.labels.data; // Labels (class ids)

    const canvas = createCanvas(originalImage.width, originalImage.height);
    const ctx = canvas.getContext('2d');

    // Draw the original image
    ctx.drawImage(originalImage, 0, 0);

    // Scaling factors to map the boxes from the 1024x1024 canvas back to the original image resolution
    const xScale = originalImage.width / 1024;
    const yScale = originalImage.height / 1024;

    // Draw bounding boxes with the correct scale
    for (let i = 0; i < boxes.length / 4; i++) {
        const score = scores[i];
        if (score > threshold) {
            crop_heads++;
            
            // Rescale the bounding box coordinates to the original image size
            const x1 = boxes[i * 4] * xScale;
            const y1 = boxes[i * 4 + 1] * yScale;
            const x2 = boxes[i * 4 + 2] * xScale;
            const y2 = boxes[i * 4 + 3] * yScale;

            // Draw bounding box and score text
            ctx.strokeStyle = 'red';
            ctx.lineWidth = Math.max(2, originalImage.width / (1024/boxLineThickness))
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            ctx.fillStyle = 'red';
            ctx.font = '16px Arial';
            ctx.fillText(`Score: ${score.toFixed(2)}`, x1, y1 > 10 ? y1 - 5 : 10);
        }
    }

    // Save the debug image
    const buffer = canvas.toBuffer('image/jpeg');
    fs.writeFileSync(path.join(BASE_PATH, "output", path.basename(imagePath)), buffer);

    return crop_heads;
}
