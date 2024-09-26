
import express from "express"
import multer from "multer"
import { exec } from "child_process"
import path from "path"
import { runCropHeadCountModel } from "./Crop-head-counter/counter.js"
import fs from "fs"
import dotenv from "dotenv"

const app = express()
dotenv.config()
const port = 1000


const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/')
    },
    filename: (req, file, cb) => {
        cb(null, new Date().getTime() + path.extname(file.originalname))
    }
})
const upload = multer({ storage: storage })


app.listen(port, () => {
    console.log("Server Started !")
})




// Function to read the image and convert it to base64
const getImageBase64 = (imagePath) => {
    return new Promise((resolve, reject) => {
        fs.readFile(imagePath, (err, data) => {
            if (err) {
                return reject(err);
            }
            const base64Image = data.toString('base64');
            resolve(base64Image);
        });
    });
};


app.post('/count-crops-python', upload.single('image'), (req, res) => {
    const imagePath = path.join(process.cwd(), 'uploads', req.file.filename)
    const debugImagePath = path.join(process.cwd(), "Crop-head-counter/output", req.file.filename)

    const pythonScriptPath = path.join(process.cwd(), 'Crop-head-counter/counter.py')

    exec(`python ${pythonScriptPath} "${imagePath}" "${path.join(process.cwd(), "Crop-head-counter")}"`, async (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${stderr}`)
            return res.status(500).send(`Error: ${stderr}`)
        }

        const count = parseInt(stdout.trim(), 10)
        const debugImage = await getImageBase64(debugImagePath);


        res.json({
            detectedCropHeads: count,
            image: `data:image/jpeg;base64,${debugImage}`
        })

        if (process.env.DELETE_CROP_HEADS_IMAGE_AFTER_UPLOAD === "false") {
            return;
        }

        fs.unlink(imagePath, (err) => { })
        fs.unlink(debugImagePath, (err) => { })
    })
})



app.post('/count-crops', upload.single('image'), async (req, res) => {
    const imagePath = path.join(process.cwd(), 'uploads', req.file.filename)
    const debugImagePath = path.join(process.cwd(), "Crop-head-counter/output", req.file.filename)

    try {
        const count = await runCropHeadCountModel(path.join(process.cwd(), "Crop-head-counter"), imagePath)
        const debugImage = await getImageBase64(debugImagePath);


        res.json({
            detectedCropHeads: count,
            image: `data:image/jpeg;base64,${debugImage}`
        })


        if (process.env.DELETE_CROP_HEADS_IMAGE_AFTER_UPLOAD === "false") {
            return;
        }

        fs.unlink(imagePath, (err) => { })
        fs.unlink(debugImagePath, (err) => { })
    } catch (err) {
        fs.unlink(imagePath, (err) => { })

        sendResponse(res, err, false);
    }
})