# InfiniteZoomFlow: Phase-Locked Fractal AI 🌀

![Fractal Zoom](https://raw.githubusercontent.com/USERNAME/REPOSITORY/main/fractalzoom.png)


> **"Not a Blind Painter, but a Lucid Dreamer."**

InfiniteZoomFlow is a real-time, physics-gated generative AI system. Unlike traditional zoomers (like Deforum) that blindly hallucinate over moving pixels, this system uses **Fractal Viscosity** to distinguish between solid objects and empty space.

It locks coherent structures in place while allowing the AI to "dream" only into the voids, creating permanent, navigable worlds rather than fever dreams.

![Demo](https://via.placeholder.com/800x450?text=Infinite+Zoom+Demo+Placeholder)

## 🚀 Key Features

* **Fractal Viscosity Gating:** A physics-based mask that prevents the AI from overwriting high-complexity textures (objects/roots/faces).
* **Infinite Zoom Engine:** Continuous affine feedback loop with sub-pixel precision.
* **Steerable Camera:** Use your mouse to fly through the generated latent space in real-time.
* **LCM Acceleration:** Uses Latent Consistency Models (SD 1.5) for high-speed (15-30 FPS) inference on consumer hardware.
* **Phase Locking:** Maintains structural integrity of the world as you fly past it.

## 🛠️ Installation

### Prerequisites
* Python 3.10 or higher
* NVIDIA GPU with 6GB+ VRAM (Recommended)

### Setup
1.  **Clone the repository:**
    ```bash
      git clone https://github.com/anttiluode/MatrixBeta.git
      cd MatrixBeta
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: For best performance, ensure you have PyTorch installed with CUDA support).*

3.  **Run the Engine:**
    ```bash
    python InfiniteZoomFlow2.py
    ```

## 🎮 Controls

* **Mouse:** Move cursor relative to the center of the window to steer the zoom direction.
* **Zoom Speed:** Controls how fast you dive into the fractal.
* **Dream Strength:** How much "new" content the AI generates per frame.
* **Coherence Gate (Viscosity):**
    * **Low (0.5):** Dreamy, fluid world. Objects morph frequently.
    * **High (1.5):** Rigid, physical world. Objects persist, only empty space is filled.

## 🧠 The Science: Why This Works

Traditional AI video generators suffer from "hallucination flicker" because they treat every pixel as equal. This system implements a **Physical Cortex** layer:

1.  **The Zoom:** The previous frame is scaled and warped to match the camera movement.
2.  **The Measurement:** The system calculates the **Fractal Dimension** (β) of every region in the image.
    * *High Complexity (High β)* = "Matter" (Don't touch).
    * *Low Complexity (Low β)* = "Void" (Generate here).
3.  **The Lock:** A spatial mask protects the "Matter" from the diffusion process.
4.  **The Dream:** The AI runs a denoising step *only* on the "Void" regions, seamlessly blending new details into the existing reality.

## 📜 License
MIT License. Feel free to modify and distribute.
