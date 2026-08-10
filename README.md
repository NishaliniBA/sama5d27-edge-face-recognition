# AI-Based Face Recognition on SAMA5D27 WLSOM EK1

Real-time, on-device face recognition running entirely on an ARM Cortex-A5 embedded Linux platform — no cloud dependency.

> Presented at **VDAT 2024**

---

## 1. Overview

This project implements an end-to-end **Edge AI face recognition system** on the **Microchip SAMA5D27 WLSOM EK1** platform. It combines a custom-built embedded Linux system with a lightweight deep learning model (**MobileFaceNet**) to perform on-device face embedding extraction and recognition, without relying on any external server or cloud service.

The work spans two layers of the embedded stack:

- **System/BSP layer** — building and integrating a custom embedded Linux image (U-Boot, Kernel, Device Tree, AI libraries) using Buildroot.
- **Application layer** — deploying an optimized TensorFlow Lite model with an OpenCV-based image processing pipeline for real-time inference.

---

## 2. Key Features

- ✅ Fully on-device inference — **no cloud dependency**
- ✅ Custom embedded Linux image built with **Buildroot**
- ✅ Lightweight **MobileFaceNet** model optimized for ARM Cortex-A5
- ✅ End-to-end pipeline: preprocessing → embedding extraction → recognition
- ✅ Focus on **memory and latency optimization** for constrained edge hardware
- ✅ Built using **Python, OpenCV, and TensorFlow Lite**

---

## 3. System Architecture

```
                ┌─────────────────────────────────────┐
                │     SAMA5D27 WLSOM EK1 (Hardware)    │
                │   ARM Cortex-A5 32-bit RISC CPU      │
                └──────────────────┬────────────────────┘
                                   │
                ┌──────────────────▼────────────────────┐
                │        Custom Embedded Linux            │
                │  U-Boot → Linux Kernel → Device Tree     │
                │           (Built via Buildroot)          │
                └──────────────────┬────────────────────┘
                                   │
                ┌──────────────────▼────────────────────┐
                │      AI Application Layer (Python)       │
                │  OpenCV (Image Capture/Preprocessing)     │
                │  TensorFlow Lite (MobileFaceNet Inference)│
                └──────────────────┬────────────────────┘
                                   │
                ┌──────────────────▼────────────────────┐
                │        Real-Time Face Recognition         │
                │        (On-Device, No Cloud)              │
                └─────────────────────────────────────────┘
```

---

## 4. Hardware

| Component        | Specification                          |
|-------------------|-----------------------------------------|
| Platform          | Microchip SAMA5D27 WLSOM EK1            |
| Processor         | 32-bit ARM Cortex-A5 RISC Processor     |

---

## 5. Software and Technologies

| Layer                  | Technology                     |
|-------------------------|---------------------------------|
| Operating System         | Embedded Linux                 |
| Linux Build System       | Buildroot                      |
| Boot Components          | U-Boot, Linux Kernel, Device Tree |
| Programming Language     | Python                         |
| Computer Vision          | OpenCV                         |
| AI Inference Framework   | TensorFlow Lite                |
| AI Model                 | MobileFaceNet                  |

---

## 6. AI/ML Pipeline

The AI pipeline is built around **MobileFaceNet**, a lightweight face embedding model suited for resource-constrained edge devices.

1. **Image Capture** — Input frame acquired via the platform's camera pipeline.
2. **Preprocessing** — Face image preprocessing performed using OpenCV.
3. **Embedding Extraction** — MobileFaceNet (via TensorFlow Lite) generates a face embedding vector from the preprocessed image.
4. **Recognition** — Extracted embeddings are used to perform real-time face recognition on-device.

> **Note:** The MobileFaceNet model was optimized for deployment on the ARM Cortex-A5 processor. Model training details/dataset are not part of this project's documented scope.

---

## 7. Embedded Linux Setup

A custom Linux image was built for the SAMA5D27 WLSOM EK1 by integrating the following components using **Buildroot**:

- **U-Boot** — Bootloader
- **Linux Kernel** — Core OS kernel
- **Device Tree** — Hardware description for the SAMA5D27 platform
- **Essential AI Libraries** — Libraries required to run OpenCV and TensorFlow Lite on-device

This custom image provides the minimal, optimized Linux environment needed to run the AI inference pipeline efficiently on constrained hardware.

---

## 8. Face Recognition Workflow

```
Camera Input
     │
     ▼
Image Preprocessing (OpenCV)
     │
     ▼
Face Embedding Extraction (MobileFaceNet / TFLite)
     │
     ▼
Real-Time Face Recognition (On-Device)
```

---

## 9. Edge AI Optimization

Optimization efforts in this project focused on:

- **Model optimization** — MobileFaceNet tuned for efficient execution on the ARM Cortex-A5 processor.
- **Memory usage optimization** — Minimizing runtime memory footprint suited for embedded constraints.
- **Latency optimization** — Reducing inference latency for real-time on-device operation.
- **Minimal Linux image** — Custom Buildroot-based image containing only essential components required for AI inference.

---

## 10. Project Structure

> ⚠️ The structure below is an **example/representative layout** for illustration purposes. Actual file/directory names were not specified in the project documentation.

```
face-recognition-sama5d27/
├── buildroot-config/          # Example: Buildroot configuration files
├── linux-kernel/              # Example: Kernel and device tree sources
├── uboot/                     # Example: U-Boot sources/config
├── models/
│   └── mobilefacenet.tflite   # Example: TFLite MobileFaceNet model
├── src/
│   ├── preprocessing.py       # Example: OpenCV preprocessing scripts
│   ├── embedding.py           # Example: Embedding extraction logic
│   └── recognition.py         # Example: Recognition pipeline
└── README.md
```

---

## 11. How It Works

1. The SAMA5D27 WLSOM EK1 boots into a custom embedded Linux image (built via Buildroot) containing U-Boot, the Linux Kernel, Device Tree, and required AI libraries.
2. A Python application, using OpenCV, captures and preprocesses image frames on-device.
3. The preprocessed face image is passed to the MobileFaceNet model running via TensorFlow Lite.
4. MobileFaceNet extracts a face embedding from the input image.
5. The system performs real-time face recognition using the extracted embedding — entirely on-device, without any cloud connection.

---

## 12. Results/Outcome

The project successfully demonstrates **real-time, on-device face recognition** on the SAMA5D27 WLSOM EK1 platform, with the AI pipeline optimized for **efficient memory usage and reduced latency** suited to edge deployment.

> No specific accuracy, FPS, or latency metrics are published as part of this documentation.

---

## 13. Applications

- Edge-based access control and identity verification
- Offline/standalone face recognition systems where cloud connectivity is unavailable or undesirable
- Embedded AI/computer vision proof-of-concept for resource-constrained ARM platforms
- Reference implementation for deploying lightweight face recognition models on Cortex-A class processors

---

## 14. Future Improvements

> The following are **proposed** directions for future work and are **not implemented** in the current project.

- Extending support to additional embedded hardware platforms
- Adding a user-facing dataset/enrollment management interface
- Formal benchmarking of accuracy, FPS, and latency
- Exploring further model compression/quantization techniques
- Adding multi-face detection and tracking support

---

## 15. Technologies Used

`ARM Cortex-A5` `Embedded Linux` `Buildroot` `U-Boot` `Linux Kernel` `Device Tree` `Python` `OpenCV` `TensorFlow Lite` `MobileFaceNet` `Edge AI`

---

## 16. Author

**NISHALINI BA**


Project presented at **VDAT 2024**

---
