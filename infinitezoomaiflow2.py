import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderKL  # <--- Added AutoencoderKL
from PIL import Image, ImageTk
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, StringVar, Entry, Canvas
from threading import Thread, Lock
import time
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # <--- Force standard downloader

# --- 1. Fractal Viscosity Engine (Spatial) ---
class FractalViscosityMap:
    def __init__(self, device):
        self.device = device
        
    def get_viscosity_mask(self, image_np, sensitivity=1.0, threshold=0.1):
        """
        Calculates a per-pixel 'structure score' (viscosity).
        Returns a mask where 1.0 = High Structure (Keep), 0.0 = Low Structure (Dream).
        """
        # Convert to grayscale float
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image_np.astype(np.float32) / 255.0
            
        # Compute Local Variance (Texture Energy)
        # Var(X) = E[X^2] - (E[X])^2
        # We use a box blur as the "Expectation" E[]
        
        sigma = 3 # Scale of structure analysis
        mean = cv2.GaussianBlur(gray, (0, 0), sigma)
        mean_sq = cv2.GaussianBlur(gray**2, (0, 0), sigma)
        variance = mean_sq - mean**2
        
        # Convert variance to "Viscosity" (0 to 1)
        # High variance = High Viscosity (Structure)
        # Low variance = Low Viscosity (Smooth)
        
        # Normalize: Sqrt(Variance) is roughly contrast. 
        # We amplify it by sensitivity.
        std_dev = np.sqrt(np.maximum(0, variance))
        viscosity = np.clip((std_dev - threshold) * sensitivity * 10.0, 0.0, 1.0)
        
        # Smooth the mask itself to avoid jagged transitions
        viscosity = cv2.GaussianBlur(viscosity, (0, 0), 2.0)
        
        # Expand to 3 channels for broadcasting
        return np.repeat(viscosity[:, :, np.newaxis], 3, axis=2)

# --- 2. Main Application: Infinite Zoom Flow ---
class InfiniteZoomFlow:
    def __init__(self, master):
        self.master = master
        self.master.title("Infinite Fractal Zoom - Phase Locked AI")
        self.master.geometry("1000x700")
        self.master.configure(bg='#111111')
        
        # Core Components
        self.frame_lock = Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}")
        
        self.viscosity_engine = FractalViscosityMap(self.device)
        self.pipe = None
        
        # State
        self.is_running = False
        self.last_frame = None  # The feedback buffer
        self.display_image = None
        
        # Zoom State
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
        
        self.params = {
            "zoom_speed": 1.05,
            "strength": 0.35,       # LCM is stronger, 0.35 is good for smooth dreaming
            "viscosity_gate": 1.2,  
            "prompt": "bioluminescent forest, fractal roots, complex machinery, 8k, detailed"
        }
        
        self.setup_gui()
        self.start_model_loading()
        
        # Bind Mouse for Steering
        self.panel.bind('<Motion>', self.update_steering)

    def setup_gui(self):
        # Layout
        main_frame = Frame(self.master, bg='#111111')
        main_frame.pack(fill='both', expand=True)
        
        # Left Sidebar
        sidebar = Frame(main_frame, width=300, bg='#222222', padx=10, pady=10)
        sidebar.pack(side='left', fill='y')
        
        Label(sidebar, text="Fractal Controls", bg='#222222', fg='white', font=("Arial", 14, "bold")).pack(pady=10)
        
        self.status_var = StringVar(value="Initializing...")
        Label(sidebar, textvariable=self.status_var, bg='#222222', fg='#00ff00').pack(pady=5)
        
        self.btn_toggle = Button(sidebar, text="Loading AI...", command=self.toggle_zoom, 
                                 bg='#444444', fg='white', state='disabled', font=("Arial", 12))
        self.btn_toggle.pack(fill='x', pady=10)
        
        # Params
        def add_slider(label, key, min_val, max_val, res, default):
            Label(sidebar, text=label, bg='#222222', fg='#aaaaaa', anchor='w').pack(fill='x', pady=(10,0))
            s = Scale(sidebar, from_=min_val, to=max_val, resolution=res, orient=HORIZONTAL,
                      bg='#222222', fg='white', troughcolor='#444444', highlightthickness=0,
                      command=lambda v: self.update_param(key, float(v)))
            s.set(default)
            s.pack(fill='x')
            
        add_slider("Zoom Speed", "zoom_speed", 1.01, 1.20, 0.01, 1.05)
        add_slider("Dream Strength", "strength", 0.1, 0.9, 0.05, 0.45)
        add_slider("Coherence Gate (Viscosity)", "viscosity_gate", 0.1, 5.0, 0.1, 1.5)
        
        Label(sidebar, text="Prompt", bg='#222222', fg='#aaaaaa', anchor='w').pack(fill='x', pady=(20,0))
        self.prompt_entry = Entry(sidebar, bg='#333333', fg='white', insertbackground='white')
        self.prompt_entry.insert(0, self.params["prompt"])
        self.prompt_entry.pack(fill='x')
        Button(sidebar, text="Update Prompt", command=self.update_prompt, bg='#555555', fg='white').pack(fill='x', pady=5)
        
        # Right Video Area
        self.panel = Label(main_frame, bg='black')
        self.panel.pack(side='right', fill='both', expand=True)

    def start_model_loading(self):
        Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            print("Loading LCM Dreamshaper (SD1.5) for Speed & Stability...")
            
            # Swapping to LCM-Dreamshaper v7 
            # This runs on SD1.5 architecture which is rock-solid on Windows
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=torch.float16,
                safety_checker=None  # Saves VRAM, prevents false positives
            ).to(self.device)
            
            # Optimization for speed
            self.pipe.set_progress_bar_config(disable=True)
            self.pipe.enable_attention_slicing() 
            
            # Initialize blank frame
            self.last_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Warmup (Compiles the CUDA graph)
            print("Warming up...")
            dummy_img = Image.fromarray(self.last_frame)
            with torch.inference_mode():
                self.pipe(
                    prompt="warmup", 
                    image=dummy_img, 
                    strength=0.5, 
                    num_inference_steps=4,  # LCM is incredibly fast (4-8 steps)
                    guidance_scale=1.0      # LCM standard guidance
                )
            
            self.master.after(0, lambda: self.status_var.set("Ready (LCM Mode)"))
            self.master.after(0, lambda: self.btn_toggle.config(state='normal', text="Start Infinite Zoom", bg='#00aa00'))
            print("Model Loaded Successfully.")
            
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
            
        except Exception as e:
            print(f"Model Load Error: {e}")
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))
    def update_param(self, key, val):
        self.params[key] = val
        
    def update_prompt(self):
        self.params["prompt"] = self.prompt_entry.get()

    def update_steering(self, event):
        # Calculate mouse position relative to center of panel
        # Normalized -0.5 to 0.5
        if self.panel.winfo_width() > 0:
            w = self.panel.winfo_width()
            h = self.panel.winfo_height()
            x = (event.x / w) 
            y = (event.y / h)
            self.zoom_center_x = np.clip(x, 0, 1)
            self.zoom_center_y = np.clip(y, 0, 1)

    def apply_affine_zoom(self, image, scale, center_x, center_y):
        """
        Zooms into the image using OpenCV affine transform.
        center_x, center_y are 0.0-1.0 normalized coordinates.
        """
        h, w = image.shape[:2]
        
        # Convert normalized center to pixels
        cx = center_x * w
        cy = center_y * h
        
        # Prepare transformation matrix
        # 1. Translate center to origin
        # 2. Scale
        # 3. Translate back
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        
        # Warp
        zoomed = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return zoomed

    def loop(self):
        while self.is_running:
            start_time = time.time()
            
            # 1. Prepare Input (The Zoom)
            with self.frame_lock:
                current_img = self.last_frame.copy()
            
            # Apply infinite zoom transform steering towards mouse
            zoomed_img = self.apply_affine_zoom(
                current_img, 
                self.params["zoom_speed"], 
                self.zoom_center_x, 
                self.zoom_center_y
            )
            
            # 2. Compute Fractal Viscosity (The "Coherence Gate")
            # Calculate where structure exists
            structure_mask = self.viscosity_engine.get_viscosity_mask(
                zoomed_img, 
                sensitivity=self.params["viscosity_gate"]
            )
            
            # 3. AI Hallucination Step
            # We ask the AI to dream on top of the zoomed image
            pil_input = Image.fromarray(zoomed_img)
            
            # Generate
            strength = self.params["strength"]
            # Dynamic strength? Maybe lower strength where structure is high?
            # For now, constant strength, mask handles the blend.
            
            with torch.inference_mode():
                dream_output = self.pipe(
                    prompt=self.params["prompt"],
                    image=pil_input,
                    strength=strength,
                    num_inference_steps=2,
                    guidance_scale=0.0
                ).images[0]
            
            dream_np = np.array(dream_output)
            
            # 4. Gated Blending (The "Phase Lock")
            # New Frame = Structure * Original_Zoomed + (1-Structure) * Dream
            # If Structure is 1.0 (High Viscosity), we keep Original (Stable)
            # If Structure is 0.0 (Smooth), we take Dream (Regenerate)
            
            final_frame = (zoomed_img * structure_mask + dream_np * (1.0 - structure_mask)).astype(np.uint8)
            
            # Update Feedback Loop
            with self.frame_lock:
                self.last_frame = final_frame
            
            # Display
            self.update_display(final_frame)
            
            # FPS Stats
            dt = time.time() - start_time
            fps = 1.0 / (dt + 1e-9)
            self.master.after(0, lambda f=fps: self.status_var.set(f"FPS: {f:.1f} | Gate Active"))

    def update_display(self, frame_np):
        # Convert to TK format
        img = Image.fromarray(frame_np)
        # Resize to fit panel if needed (optional, purely visual)
        # For now display 1:1 or centered
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Safe TK update
        def _update():
            self.panel.configure(image=imgtk)
            self.panel.image = imgtk # Keep ref
        
        self.master.after(0, _update)

    def toggle_zoom(self):
        if self.is_running:
            self.is_running = False
            self.btn_toggle.config(text="Start Infinite Zoom", bg='#00aa00')
        else:
            self.is_running = True
            self.btn_toggle.config(text="Stop", bg='#aa0000')
            Thread(target=self.loop, daemon=True).start()

if __name__ == "__main__":
    root = Tk()
    app = InfiniteZoomFlow(root)
    root.mainloop()