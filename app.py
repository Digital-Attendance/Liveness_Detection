import json
import base64
import cv2
import numpy as np
import asyncio
from operation import detect
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=6)


async def process_image(encoded_image):
    try:
        loop = asyncio.get_running_loop()
        image_bytes = await loop.run_in_executor(
            executor, base64.b64decode, encoded_image
        )
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = await loop.run_in_executor(
            executor, lambda: cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        )
        if image is None:
            raise ValueError(
                "Failed to decode image. The input may be corrupted or in an unsupported format."
            )
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


async def detect_liveness(image):
    try:
        loop = asyncio.get_running_loop()
        label, value = await loop.run_in_executor(executor, detect, image)
        label = int(label) if isinstance(label, (np.integer, np.int64)) else label
        value = float(value) if isinstance(value, (np.floating, np.float64)) else value
        return {"label": label, "confidence": value}
    except Exception as e:
        raise ValueError(f"Error in liveness detection: {e}")


async def async_lambda_handler(event, context):
    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)
        if "image" not in body:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "Missing 'image' field in the request body."}
                ),
            }
        encoded_image = body["image"]
        image = await process_image(encoded_image)
        result = await detect_liveness(image)
        return {"statusCode": 200, "body": json.dumps(result)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def lambda_handler(event, context):
    return asyncio.run(async_lambda_handler(event, context))
