from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from test import islive
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers=4)


@app.route("/")
def index():
    return render_template("index.html")


# Asynchronous function to process the image
async def process_image(file):
    loop = asyncio.get_event_loop()
    encoded_image = await loop.run_in_executor(executor, file.read)
    nparr = np.frombuffer(encoded_image, np.uint8)
    image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    label, value = await loop.run_in_executor(executor, islive, image_np)
    label = int(label) if isinstance(label, (np.integer, np.int64)) else label
    value = float(value) if isinstance(value, (np.floating, np.float64)) else value

    if label == 2:
        label = 0

    return {"label": label, "confidence": value}


@app.route("/detect_liveness", methods=["POST"])
def detect_liveness():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"})

    file = request.files["image"]

    # Run the image processing asynchronously
    result = asyncio.run(process_image(file))
    print(result)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
