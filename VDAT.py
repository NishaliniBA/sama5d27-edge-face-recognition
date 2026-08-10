```python
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util


# ============================================================
# Function to print detection results
# ============================================================

def print_output(message):
    print(message)


# ============================================================
# Webcam Video Stream
# ============================================================

class VideoStream:
    """Camera object that controls video streaming from the webcam."""

    def __init__(self, resolution=(640, 480), framerate=30):

        # Initialize camera
        self.stream = cv2.VideoCapture(0)

        # Set camera format
        self.stream.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

        # Set resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        # Set requested frame rate
        self.stream.set(cv2.CAP_PROP_FPS, framerate)

        # Check if camera opened successfully
        if not self.stream.isOpened():
            print("Error: Could not open video device.")
            raise RuntimeError("Camera initialization failed.")

        # Capture first frame
        self.grabbed, self.frame = self.stream.read()

        # Check first frame
        if (
            not self.grabbed
            or self.frame is None
            or self.frame.size == 0
        ):
            print("Error: Frame capture failed or the frame is empty.")
            self.stopped = True
        else:
            self.stopped = False

    def start(self):
        """Start the thread that continuously captures frames."""

        Thread(
            target=self.update,
            args=(),
            daemon=True
        ).start()

        return self

    def update(self):
        """Continuously capture frames from the webcam."""

        while not self.stopped:

            self.grabbed, self.frame = self.stream.read()

            if (
                not self.grabbed
                or self.frame is None
                or self.frame.size == 0
            ):
                print(
                    "Error: Frame capture failed or the frame is empty."
                )
                self.stopped = True
                break

        self.stream.release()

    def read(self):
        """Return the latest captured frame."""
        return self.frame

    def stop(self):
        """Stop the video stream."""
        self.stopped = True


# ============================================================
# Command Line Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--modeldir",
    help="Folder where the .tflite model is located",
    required=True
)

parser.add_argument(
    "--graph",
    help="Name of the .tflite model file",
    default="detect.tflite"
)

parser.add_argument(
    "--labels",
    help="Name of the labelmap file",
    default="labelmap.txt"
)

parser.add_argument(
    "--threshold",
    help="Minimum confidence threshold for displaying detections",
    default=0.5
)

parser.add_argument(
    "--resolution",
    help="Desired webcam resolution in WxH",
    default="640x480"
)

parser.add_argument(
    "--edgetpu",
    help="Use Coral Edge TPU Accelerator",
    action="store_true"
)

args = parser.parse_args()


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)

try:
    resW, resH = args.resolution.split("x")
    imW, imH = int(resW), int(resH)
except ValueError:
    raise ValueError(
        "Invalid resolution format. Use WIDTHxHEIGHT, "
        "for example: 640x480"
    )

use_TPU = args.edgetpu


# ============================================================
# Load TensorFlow Lite Interpreter
# ============================================================

pkg = importlib.util.find_spec("tflite_runtime")

if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate

else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


# ============================================================
# Edge TPU Model
# ============================================================

if use_TPU and GRAPH_NAME == "detect.tflite":
    GRAPH_NAME = "edgetpu.tflite"


# ============================================================
# Model Paths
# ============================================================

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(
    CWD_PATH,
    MODEL_NAME,
    GRAPH_NAME
)

PATH_TO_LABELS = os.path.join(
    CWD_PATH,
    MODEL_NAME,
    LABELMAP_NAME
)


# ============================================================
# Validate Model Files
# ============================================================

if not os.path.isfile(PATH_TO_CKPT):
    raise FileNotFoundError(
        f"TensorFlow Lite model not found: {PATH_TO_CKPT}"
    )

if not os.path.isfile(PATH_TO_LABELS):
    raise FileNotFoundError(
        f"Label map not found: {PATH_TO_LABELS}"
    )


# ============================================================
# Load Label Map
# ============================================================

with open(PATH_TO_LABELS, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

if labels and labels[0] == "???":
    del labels[0]


# ============================================================
# Load TensorFlow Lite Model
# ============================================================

if use_TPU:
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT,
        experimental_delegates=[
            load_delegate("libedgetpu.so.1.0")
        ]
    )
else:
    interpreter = Interpreter(
        model_path=PATH_TO_CKPT
    )


# ============================================================
# Allocate Model Tensors
# ============================================================

interpreter.allocate_tensors()


# ============================================================
# Get Model Input / Output Details
# ============================================================

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]["shape"][1]
width = input_details[0]["shape"][2]

floating_model = (
    input_details[0]["dtype"] == np.float32
)

input_mean = 127.5
input_std = 127.5


# ============================================================
# Initialize Video Stream
# ============================================================

videostream = VideoStream(
    resolution=(imW, imH),
    framerate=30
).start()

time.sleep(1)


# ============================================================
# Main Detection Loop
# ============================================================

try:

    while True:

        # Start timer
        t1 = cv2.getTickCount()

        # Read frame
        frame1 = videostream.read()

        # Check frame
        if (
            frame1 is None
            or frame1.size == 0
        ):
            print(
                "Error: Frame capture failed or "
                "the frame is empty."
            )
            break

        # ----------------------------------------------------
        # Convert BGR to RGB
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame1,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Resize frame to model input dimensions
        # ----------------------------------------------------

        frame_resized = cv2.resize(
            frame_rgb,
            (width, height)
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        input_data = np.expand_dims(
            frame_resized,
            axis=0
        )

        # ----------------------------------------------------
        # Normalize floating-point model input
        # ----------------------------------------------------

        if floating_model:

            input_data = (
                np.float32(input_data)
                - input_mean
            ) / input_std

        # ----------------------------------------------------
        # Run TensorFlow Lite inference
        # ----------------------------------------------------

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        # ----------------------------------------------------
        # Get detection results
        # ----------------------------------------------------

        boxes = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        classes = interpreter.get_tensor(
            output_details[1]["index"]
        )[0]

        scores = interpreter.get_tensor(
            output_details[2]["index"]
        )[0]

        # ----------------------------------------------------
        # Process detection results
        # ----------------------------------------------------

        for i in range(len(scores)):

            if (
                scores[i] > min_conf_threshold
                and scores[i] <= 1.0
            ):

                class_id = int(classes[i])

                if class_id < len(labels):

                    detected_class = labels[class_id]

                    if detected_class.lower() == "person":

                        print_output(
                            "Person detected"
                        )

        # Small delay
        time.sleep(0.1)

except KeyboardInterrupt:

    print("\nStopping detection...")

finally:

    videostream.stop()
```
