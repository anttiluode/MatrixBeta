#!/usr/bin/env python3
"""
MATRIX VISION: Multi-Scale β-Gradient (CORRECTED)
==================================================

Restores the ORIGINAL multi-scale fractal measurement that was lost
in the "GPU optimization". 

The darkness was an artifact of single-scale oversimplification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import AutoPipelineForImage2Image
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class GPUEngine:
    """
    CORRECTED: Multi-scale β-gradient computation on GPU.
    """
    
    def __init__(self, device='cuda', dtype=torch.float16):
        self.device = device
        self.dtype = dtype
        
        # CRITICAL FIX: Create Sobel kernels with correct dtype
        self.sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
            dtype=dtype,  # Match the working dtype
            device=device
        ).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
            dtype=dtype,  # Match the working dtype
            device=device
        ).view(1, 1, 3, 3)
        
        # MULTI-SCALE Gaussian kernels (THE KEY FIX)
        self.blur_levels = [0, 2, 4, 8]
        self.blur_kernels = {}
        
        for radius in self.blur_levels:
            if radius > 0:
                kernel = self._create_gaussian_kernel(radius)
                # Convert to working dtype
                self.blur_kernels[radius] = kernel.to(device).to(dtype)
    
    def _create_gaussian_kernel(self, sigma, kernel_size=None):
        """Create 2D Gaussian blur kernel."""
        if kernel_size is None:
            kernel_size = int(2 * 4 * sigma + 1)
        
        # Create coordinate grids
        x = torch.arange(kernel_size, dtype=torch.float32)
        x_grid = x.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.
        
        # Gaussian formula
        gaussian = (1. / (2. * np.pi * variance)) * \
                   torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
        
        gaussian = gaussian / torch.sum(gaussian)
        return gaussian.view(1, 1, kernel_size, kernel_size)
    
    def _apply_blur(self, tensor, radius):
        """Apply Gaussian blur at specified radius."""
        if radius == 0:
            return tensor
        
        kernel = self.blur_kernels[radius]
        padding = kernel.shape[-1] // 2
        
        # Apply to each channel separately
        b, c, h, w = tensor.shape
        return F.conv2d(
            tensor, 
            kernel.expand(c, 1, -1, -1), 
            padding=padding, 
            groups=c
        )
    
    def _compute_gradient_magnitude(self, tensor):
        """Compute gradient magnitude using Sobel."""
        b, c, h, w = tensor.shape
        
        gx = F.conv2d(tensor, self.sobel_x.expand(c, 1, -1, -1), padding=1, groups=c)
        gy = F.conv2d(tensor, self.sobel_y.expand(c, 1, -1, -1), padding=1, groups=c)
        
        return torch.sqrt(gx**2 + gy**2 + 1e-8)
    
    def compute_viscosity_mask(self, img_tensor, sensitivity=1.0):
        """
        CORRECTED: Multi-scale β-gradient computation.
        
        This is the ORIGINAL method that was lost in simplification.
        """
        # Convert to grayscale for structure analysis
        gray = 0.299 * img_tensor[:, 0:1] + \
               0.587 * img_tensor[:, 1:2] + \
               0.114 * img_tensor[:, 2:3]
        
        # Measure complexity at each blur level
        complexities = []
        
        for radius in self.blur_levels:
            # Apply blur
            blurred = self._apply_blur(gray, radius)
            
            # Measure gradient magnitude at this scale
            complexity = self._compute_gradient_magnitude(blurred)
            
            # Average over spatial dimensions to get single value per pixel
            complexities.append(complexity)
        
        # THE KEY: β = How much structure SURVIVES blur
        # High β (structure persists) → High viscosity (preserve)
        # Low β (structure dies) → Low viscosity (regenerate)
        
        beta = complexities[0] - complexities[-1]  # Difference: no blur vs. max blur
        
        # Normalize β to [0, 1] range for viscosity
        # Positive β = structure survives = high viscosity
        # Negative or zero β = structure dies = low viscosity
        
        # Adaptive threshold based on image statistics
        threshold = torch.mean(beta) * 0.5
        
        viscosity = torch.sigmoid((beta - threshold) * 10.0 * sensitivity)
        
        # Expand to 3 channels
        return viscosity.repeat(1, 3, 1, 1)
    
    def affine_zoom(self, img_tensor, zoom_factor, center_x, center_y):
        """GPU-accelerated affine zoom with bicubic interpolation."""
        if img_tensor is None:
            return None
        
        B, C, H, W = img_tensor.shape
        
        # Store input dtype for later conversion
        input_dtype = img_tensor.dtype
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        
        # Zoom center (normalized to [-1, 1])
        cx = (center_x * 2) - 1
        cy = (center_y * 2) - 1
        
        # Apply zoom transformation
        grid_x = (grid_x - cx) / zoom_factor + cx
        grid_y = (grid_y - cy) / zoom_factor + cy
        
        # Stack into grid [1, H, W, 2]
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        
        # CRITICAL FIX: Match grid dtype to input tensor
        grid = grid.to(input_dtype)
        
        # Sample with bicubic interpolation
        return F.grid_sample(
            img_tensor, 
            grid, 
            mode='bicubic', 
            padding_mode='reflection', 
            align_corners=True
        )


class MatrixVisionApp:
    """Main application with corrected multi-scale β."""
    
    def __init__(self, master):
        self.master = master
        self.master.title("Matrix Vision: Multi-Scale β-Gradient (CORRECTED)")
        self.master.geometry("1200x800")
        self.master.configure(bg='#050505')
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"⚡ Powering up on {self.device} with dtype {self.dtype}...")
        
        # Initialize engine with corrected multi-scale β
        self.engine = GPUEngine(self.device, self.dtype)
        
        self.pipe = None
        self.running = False
        
        # State
        self.current_tensor = None
        
        # Parameters
        self.params = {
            "zoom": 1.02,
            "strength": 0.4,
            "viscosity": 1.0,
            "prompt": "beautiful mountain landscape, sunset, 8k",  # Changed from cyberpunk to avoid NSFW filter
            "show_mask": False
        }
        
        # Mouse position
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        
        self.setup_ui()
        self.start_loader()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Sidebar
        panel = tk.Frame(self.master, bg='#111', width=300)
        panel.pack(side='left', fill='y')
        
        # Title
        tk.Label(
            panel, 
            text="MATRIX VISION", 
            font=("Consolas", 16, "bold"), 
            fg="#0f0", 
            bg="#111"
        ).pack(pady=20)
        
        # Subtitle
        tk.Label(
            panel,
            text="Multi-Scale β-Gradient",
            font=("Consolas", 10),
            fg="#888",
            bg="#111"
        ).pack()
        
        # Status
        self.status = tk.StringVar(value="Loading AI model...")
        tk.Label(
            panel, 
            textvariable=self.status, 
            fg="#888", 
            bg="#111"
        ).pack(pady=5)
        
        # Sliders
        self.add_slider(panel, "Zoom Speed", "zoom", 1.0, 1.1, 0.001)
        self.add_slider(panel, "Dream Strength", "strength", 0.1, 0.9, 0.05)
        self.add_slider(panel, "Structure Gate", "viscosity", 0.1, 5.0, 0.1)
        
        # Prompt
        tk.Label(panel, text="Prompt", fg="#888", bg="#111").pack(pady=(20,0))
        self.prompt_box = tk.Entry(panel, bg="#222", fg="white", insertbackground="white")
        self.prompt_box.insert(0, self.params["prompt"])
        self.prompt_box.pack(fill='x', padx=10)
        
        tk.Button(
            panel, 
            text="Update Prompt", 
            command=self.update_prompt, 
            bg="#333", 
            fg="white"
        ).pack(fill='x', padx=10, pady=5)
        
        # Toggle button (initially disabled)
        self.btn_toggle = tk.Button(
            panel, 
            text="LOADING...", 
            command=self.toggle,
            bg="#222", 
            fg="#888", 
            font=("Arial", 12, "bold"),
            state="disabled"
        )
        self.btn_toggle.pack(pady=20, fill='x', padx=10)
        
        # Mask toggle
        tk.Button(
            panel, 
            text="Toggle Viscosity View", 
            command=self.toggle_mask,
            bg="#222", 
            fg="#aaa"
        ).pack(fill='x', padx=10)
        
        # Info text
        info_text = (
            "CORRECTED:\n"
            "Now using multi-scale\n"
            "β-gradient measurement.\n\n"
            "Darkness should no longer\n"
            "accumulate as artifact."
        )
        tk.Label(
            panel,
            text=info_text,
            fg="#0a0",
            bg="#111",
            font=("Consolas", 8),
            justify="left"
        ).pack(pady=20, padx=10)
        
        # Viewport
        self.canvas = tk.Label(self.master, bg="black")
        self.canvas.pack(side='right', fill='both', expand=True)
        self.canvas.bind('<Motion>', self.on_mouse)
    
    def add_slider(self, parent, label, key, min_v, max_v, res):
        """Add a labeled slider."""
        tk.Label(parent, text=label, fg="#aaa", bg="#111").pack(
            anchor='w', padx=10, pady=(10,0)
        )
        
        s = tk.Scale(
            parent, 
            from_=min_v, 
            to=max_v, 
            resolution=res,
            orient='horizontal',
            bg="#111", 
            fg="white", 
            troughcolor="#333",
            highlightthickness=0,
            command=lambda v: self.params.update({key: float(v)})
        )
        s.set(self.params[key])
        s.pack(fill='x', padx=10)
    
    def start_loader(self):
        """Start model loading in background."""
        Thread(target=self.load_ai, daemon=True).start()
    
    def load_ai(self):
        """Load the diffusion model."""
        try:
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=self.dtype
            ).to(self.device)
            
            self.pipe.set_progress_bar_config(disable=True)
            
            # Create initial seed
            self.current_tensor = torch.rand(
                (1, 3, 512, 512), 
                device=self.device, 
                dtype=self.dtype
            )
            
            # Warmup
            warmup_img = self.tensor_to_pil(self.current_tensor)
            self.pipe(
                prompt="test",
                image=warmup_img,
                num_inference_steps=1,
                strength=0.3
            )
            
            # Enable UI
            self.master.after(0, lambda: self.status.set("✓ System Online"))
            self.master.after(0, lambda: self.btn_toggle.config(
                text="START ENGINE",
                bg="#004400",
                fg="#0f0",
                state='normal'
            ))
            
        except Exception as e:
            self.master.after(0, lambda: self.status.set(f"Error: {e}"))
    
    def update_prompt(self):
        """Update the generation prompt."""
        self.params["prompt"] = self.prompt_box.get()
    
    def toggle_mask(self):
        """Toggle viscosity mask visualization."""
        self.params["show_mask"] = not self.params["show_mask"]
    
    def toggle(self):
        """Start/stop the generation loop."""
        if self.running:
            self.running = False
            self.btn_toggle.config(text="START ENGINE", bg="#004400")
        else:
            if self.current_tensor is None:
                return
            self.running = True
            self.btn_toggle.config(text="STOP ENGINE", bg="#440000")
            Thread(target=self.loop, daemon=True).start()
    
    def on_mouse(self, event):
        """Track mouse position for zoom center."""
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 0 and h > 0:
            self.mouse_x = event.x / w
            self.mouse_y = event.y / h
    
    def loop(self):
        """Main generation loop."""
        while self.running:
            if self.current_tensor is None:
                time.sleep(0.1)
                continue
            
            t0 = time.perf_counter()
            
            # 1. Zoom
            zoomed = self.engine.affine_zoom(
                self.current_tensor,
                self.params["zoom"],
                self.mouse_x,
                self.mouse_y
            )
            
            # 2. Compute MULTI-SCALE β viscosity (CORRECTED)
            mask = self.engine.compute_viscosity_mask(
                zoomed, 
                self.params["viscosity"]
            )
            
            # 3. Generate
            input_pil = self.tensor_to_pil(zoomed)
            
            with torch.inference_mode():
                output = self.pipe(
                    prompt=self.params["prompt"],
                    image=input_pil,
                    strength=self.params["strength"],
                    num_inference_steps=4,
                    guidance_scale=1.0
                ).images[0]
            
            dream_tensor = self.pil_to_tensor(output)
            
            # 4. Phase lock (apply viscosity mask)
            next_frame = (zoomed * mask) + (dream_tensor * (1.0 - mask))
            
            self.current_tensor = next_frame
            
            # 5. Display
            display = mask if self.params["show_mask"] else next_frame
            self.update_display(self.tensor_to_pil(display))
            
            # Stats
            fps = 1.0 / (time.perf_counter() - t0)
            self.master.after(0, lambda f=fps: self.status.set(
                f"FPS: {f:.1f} | Multi-Scale β Active"
            ))
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image."""
        img = tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def pil_to_tensor(self, pil_img):
        """Convert PIL Image to tensor."""
        arr = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device).to(self.dtype)
    
    def update_display(self, img):
        """Update canvas with new image."""
        tk_img = ImageTk.PhotoImage(img.resize((800, 800)))
        self.master.after(0, lambda: self.canvas.configure(image=tk_img))
        self.master.after(0, lambda: setattr(self.canvas, 'image', tk_img))


if __name__ == "__main__":
    root = tk.Tk()
    app = MatrixVisionApp(root)
    root.mainloop()