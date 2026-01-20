"""
INFINITE ZOOM HIVEMIND: Inter-Model Communication for Stable Generation
========================================================================

Architecture:
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   VISION BRAIN  │────▶│  MEMORY BRAIN   │────▶│   MOTOR BRAIN   │
│   (CLIP)        │     │  (Embedding     │     │   (Diffusion)   │
│   "What IS"     │     │   Buffer)       │     │   "Make it"     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         └──────────────────────┴───────────────────────┘
                    SHARED EMBEDDING SPACE
                    
Key Innovation: The diffusion model receives SEMANTICALLY GROUNDED conditioning
that combines:
1. Scene understanding (what's currently visible)
2. Boundary context (what's at the edges of voids)  
3. Temporal memory (what the world has been)
4. User intent (the prompt, but weighted appropriately)

This prevents semantic drift and maintains object coherence across frames.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageTk
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, StringVar, Entry, Checkbutton, IntVar, Canvas, Scrollbar
from threading import Thread, Lock
import time
import os
from collections import deque

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


# =============================================================================
# 1. FRACTAL VISCOSITY ENGINE (Spatial Gating - unchanged)
# =============================================================================
class FractalViscosityMap:
    """Identifies structure vs void regions for selective dreaming."""
    
    def __init__(self, device):
        self.device = device
        
    def get_viscosity_mask(self, image_np, sensitivity=1.0, threshold=0.1):
        """
        Returns mask: 1.0 = High Structure (Keep), 0.0 = Low Structure (Dream).
        """
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image_np.astype(np.float32) / 255.0
            
        sigma = 3
        mean = cv2.GaussianBlur(gray, (0, 0), sigma)
        mean_sq = cv2.GaussianBlur(gray**2, (0, 0), sigma)
        variance = mean_sq - mean**2
        
        std_dev = np.sqrt(np.maximum(0, variance))
        viscosity = np.clip((std_dev - threshold) * sensitivity * 10.0, 0.0, 1.0)
        viscosity = cv2.GaussianBlur(viscosity, (0, 0), 2.0)
        
        return np.repeat(viscosity[:, :, np.newaxis], 3, axis=2)


# =============================================================================
# 2. SEMANTIC STEERING ENGINE (The Inter-Model Bridge)
# =============================================================================
class SemanticSteeringEngine:
    """
    The HIVEMIND: Coordinates vision understanding with generation.
    
    This is where the magic happens - we use CLIP (which shares embedding space
    with Stable Diffusion) to understand the scene and steer generation.
    """
    
    def __init__(self, device, memory_length=16):
        self.device = device
        self.memory_length = memory_length
        
        print("Loading CLIP Vision Brain...")
        # CLIP ViT-L/14 - same as SD 1.5 uses for conditioning
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.float16
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # Freeze CLIP - it's our oracle, not a learner
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()
        
        # Memory buffers
        self.scene_memory = deque(maxlen=memory_length)  # Global scene embeddings
        self.boundary_memory = deque(maxlen=memory_length // 2)  # Edge context
        
        # Cached prompt embedding
        self.cached_prompt_emb = None
        self.cached_prompt_text = None
        
        # Statistics for debugging
        self.stats = {
            'scene_drift': 0.0,
            'boundary_similarity': 0.0,
            'memory_coherence': 0.0
        }
        
        print("Semantic Steering Engine Ready.")
    
    def encode_image(self, image_np):
        """Encode full image to CLIP embedding."""
        pil_img = Image.fromarray(image_np)
        inputs = self.clip_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.to(torch.float16)
    
    def encode_text(self, text):
        """Encode text prompt to CLIP embedding."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.to(torch.float16)
    
    def analyze_boundary_context(self, image_np, viscosity_mask):
        """
        Extract semantic understanding of what's at void boundaries.
        This tells the diffusion model what should CONTINUE into voids.
        """
        # Get binary structure mask
        structure_binary = (viscosity_mask[:, :, 0] > 0.5).astype(np.uint8)
        
        # Find boundary regions (dilate structure, subtract original)
        kernel = np.ones((21, 21), np.uint8)
        dilated = cv2.dilate(structure_binary, kernel, iterations=1)
        boundary = dilated - structure_binary
        
        # Find void regions (inverse of structure)
        void_mask = 1 - structure_binary
        void_dilated = cv2.dilate(void_mask, kernel, iterations=1)
        void_boundary = void_dilated - void_mask
        
        # Combine: regions where structure meets void
        transition_zone = cv2.bitwise_and(boundary, void_boundary)
        
        if transition_zone.sum() < 500:
            return None
        
        # Sample patches from transition zones
        ys, xs = np.where(transition_zone > 0)
        if len(ys) < 10:
            return None
        
        # Sample multiple patches and encode them
        patch_embeddings = []
        n_patches = min(8, len(ys) // 10)
        
        indices = np.random.choice(len(ys), n_patches, replace=False)
        
        for idx in indices:
            y, x = ys[idx], xs[idx]
            
            # Extract 64x64 patch centered on boundary point
            h, w = image_np.shape[:2]
            y1, y2 = max(0, y-32), min(h, y+32)
            x1, x2 = max(0, x-32), min(w, x+32)
            
            patch = image_np[y1:y2, x1:x2]
            
            if patch.shape[0] >= 32 and patch.shape[1] >= 32:
                # Resize to CLIP input size
                patch_pil = Image.fromarray(patch).resize((224, 224), Image.BILINEAR)
                
                inputs = self.clip_processor(images=patch_pil, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    feat = self.clip_model.get_image_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    patch_embeddings.append(feat)
        
        if len(patch_embeddings) == 0:
            return None
        
        # Average all boundary patches
        boundary_emb = torch.cat(patch_embeddings, dim=0).mean(dim=0, keepdim=True)
        boundary_emb = boundary_emb / boundary_emb.norm(dim=-1, keepdim=True)
        
        return boundary_emb.to(torch.float16)
    
    def compute_steering_embedding(self, image_np, viscosity_mask, prompt, 
                                    alpha_scene=0.35, alpha_boundary=0.25, 
                                    alpha_memory=0.20, alpha_prompt=0.20):
        """
        THE CORE ALGORITHM: Blend multiple semantic signals into unified steering.
        
        Args:
            image_np: Current frame
            viscosity_mask: Structure mask
            prompt: User text prompt
            alpha_*: Blending weights (should sum to 1.0)
        
        Returns:
            steering_emb: [1, 768] CLIP embedding for conditioning
        """
        
        # 1. SCENE UNDERSTANDING: What IS the current image?
        scene_emb = self.encode_image(image_np)
        
        # 2. BOUNDARY CONTEXT: What's at the edges of voids?
        boundary_emb = self.analyze_boundary_context(image_np, viscosity_mask)
        
        # 3. PROMPT INTENT: What does the user want?
        if prompt != self.cached_prompt_text:
            self.cached_prompt_emb = self.encode_text(prompt)
            self.cached_prompt_text = prompt
        prompt_emb = self.cached_prompt_emb
        
        # 4. TEMPORAL MEMORY: What has this world been?
        self.scene_memory.append(scene_emb)
        if boundary_emb is not None:
            self.boundary_memory.append(boundary_emb)
        
        if len(self.scene_memory) > 1:
            memory_emb = torch.cat(list(self.scene_memory), dim=0).mean(dim=0, keepdim=True)
            memory_emb = memory_emb / memory_emb.norm(dim=-1, keepdim=True)
        else:
            memory_emb = scene_emb
        
        # 5. BLEND: Weighted combination in embedding space
        # Renormalize weights if boundary is missing
        if boundary_emb is None:
            total = alpha_scene + alpha_memory + alpha_prompt
            alpha_scene_adj = alpha_scene / total
            alpha_memory_adj = alpha_memory / total
            alpha_prompt_adj = alpha_prompt / total
            
            steering = (alpha_scene_adj * scene_emb + 
                       alpha_memory_adj * memory_emb + 
                       alpha_prompt_adj * prompt_emb)
        else:
            steering = (alpha_scene * scene_emb + 
                       alpha_boundary * boundary_emb +
                       alpha_memory * memory_emb + 
                       alpha_prompt * prompt_emb)
        
        # Renormalize to unit sphere
        steering = steering / steering.norm(dim=-1, keepdim=True)
        
        # Update statistics
        if len(self.scene_memory) > 1:
            prev_scene = list(self.scene_memory)[-2]
            self.stats['scene_drift'] = 1.0 - F.cosine_similarity(scene_emb, prev_scene).item()
            self.stats['memory_coherence'] = F.cosine_similarity(scene_emb, memory_emb).item()
        
        if boundary_emb is not None:
            self.stats['boundary_similarity'] = F.cosine_similarity(scene_emb, boundary_emb).item()
        
        return steering
    
    def get_stats_string(self):
        """Return readable stats for GUI."""
        return (f"Drift: {self.stats['scene_drift']:.3f} | "
                f"Coherence: {self.stats['memory_coherence']:.3f}")


# =============================================================================
# 3. EMBEDDING INJECTION MODULE (Bridge to Diffusion)
# =============================================================================
class EmbeddingInjector:
    """
    Converts CLIP image embeddings to SD text encoder space.
    
    SD1.5 uses CLIP text encoder, but the text and image encoders have
    different architectures. We need a projection layer.
    
    Simple approach: Use the CLIP text encoder to create a "base" embedding
    from the prompt, then BLEND with the image embedding in CLIP space
    before projecting.
    """
    
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        
        # The key insight: SD's text encoder outputs [batch, seq_len, hidden_dim]
        # CLIP image features are [batch, hidden_dim]
        # We need to expand and blend appropriately
        
    def create_conditioning(self, steering_emb, prompt, blend_strength=0.7):
        """
        Create conditioning tensor for diffusion model.
        
        Args:
            steering_emb: [1, 768] from SemanticSteeringEngine
            prompt: Text prompt for base conditioning
            blend_strength: How much steering vs pure text (0=text only, 1=steering only)
        
        Returns:
            prompt_embeds: Conditioning tensor for SD
        """
        # Get text conditioning (the normal SD path)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]  # [1, 77, 768]
        
        # Now blend: We modify the text embeddings by adding the steering signal
        # Strategy: Add steering_emb to each position, weighted
        
        # Expand steering to match sequence length
        steering_expanded = steering_emb.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        steering_expanded = steering_expanded.to(text_embeddings.dtype)
        
        # Blend
        blended = (1 - blend_strength) * text_embeddings + blend_strength * steering_expanded
        
        # Renormalize per position (helps stability)
        blended = blended / blended.norm(dim=-1, keepdim=True) * text_embeddings.norm(dim=-1, keepdim=True)
        
        return blended


# =============================================================================
# 4. MAIN APPLICATION: HIVEMIND ZOOM
# =============================================================================
class InfiniteZoomHivemind:
    """
    The complete system with inter-model communication.
    """
    
    def __init__(self, master):
        self.master = master
        self.master.title("Infinite Zoom HIVEMIND - Inter-Model Communication")
        self.master.geometry("1100x750")
        self.master.configure(bg='#111111')
        
        # Core state
        self.frame_lock = Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device}")
        
        # Subsystems (initialized during loading)
        self.viscosity_engine = FractalViscosityMap(self.device)
        self.steering_engine = None
        self.embedding_injector = None
        self.pipe = None
        
        # State
        self.is_running = False
        self.last_frame = None
        self.use_hivemind = True  # Toggle for A/B testing
        
        # Zoom state
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
        
        self.params = {
            "zoom_speed": 1.05,
            "strength": 0.40,
            "viscosity_gate": 1.5,
            "semantic_blend": 0.6,  # How much steering vs raw prompt
            "alpha_scene": 0.35,
            "alpha_boundary": 0.25,
            "alpha_memory": 0.20,
            "alpha_prompt": 0.20,
            "prompt": "bioluminescent forest, fractal roots, complex machinery, 8k, detailed"
        }
        
        self.setup_gui()
        self.start_model_loading()
        self.panel.bind('<Motion>', self.update_mouse_steering)

    def setup_gui(self):
        main_frame = Frame(self.master, bg='#111111')
        main_frame.pack(fill='both', expand=True)
        
        # Left Sidebar Container with Scrollbar
        sidebar_container = Frame(main_frame, width=320, bg='#222222')
        sidebar_container.pack(side='left', fill='y')
        sidebar_container.pack_propagate(False)
        
        # Canvas for scrolling
        canvas = Canvas(sidebar_container, bg='#222222', highlightthickness=0, width=300)
        scrollbar = Scrollbar(sidebar_container, orient="vertical", command=canvas.yview)
        
        # Scrollable frame inside canvas
        sidebar = Frame(canvas, bg='#222222', padx=10, pady=10)
        
        # Configure scroll region when sidebar changes size
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        sidebar.bind('<Configure>', configure_scroll)
        
        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=sidebar, anchor="nw")
        
        # Make canvas window expand to canvas width
        def configure_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', configure_canvas)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # === SIDEBAR CONTENTS ===
        
        Label(sidebar, text="HIVEMIND Controls", bg='#222222', fg='white', 
              font=("Arial", 14, "bold")).pack(pady=10)
        
        self.status_var = StringVar(value="Initializing...")
        Label(sidebar, textvariable=self.status_var, bg='#222222', fg='#00ff00', 
              wraplength=260).pack(pady=5)
        
        self.stats_var = StringVar(value="")
        Label(sidebar, textvariable=self.stats_var, bg='#222222', fg='#00aaff',
              font=("Arial", 9)).pack(pady=2)
        
        self.btn_toggle = Button(sidebar, text="Loading...", command=self.toggle_zoom,
                                 bg='#444444', fg='white', state='disabled', 
                                 font=("Arial", 12))
        self.btn_toggle.pack(fill='x', pady=10)
        
        # Hivemind toggle
        self.hivemind_var = IntVar(value=1)
        Checkbutton(sidebar, text="Enable Hivemind (Semantic Steering)", 
                   variable=self.hivemind_var, bg='#222222', fg='white',
                   selectcolor='#444444', activebackground='#222222',
                   command=self.toggle_hivemind).pack(fill='x', pady=5)
        
        # Sliders
        def add_slider(label, key, min_val, max_val, res, default):
            Label(sidebar, text=label, bg='#222222', fg='#aaaaaa', anchor='w').pack(fill='x', pady=(6,0))
            s = Scale(sidebar, from_=min_val, to=max_val, resolution=res, orient=HORIZONTAL,
                      bg='#222222', fg='white', troughcolor='#444444', highlightthickness=0,
                      length=250, command=lambda v: self.update_param(key, float(v)))
            s.set(default)
            s.pack(fill='x')
        
        add_slider("Zoom Speed", "zoom_speed", 1.01, 1.20, 0.01, 1.05)
        add_slider("Dream Strength", "strength", 0.1, 0.9, 0.05, 0.40)
        add_slider("Viscosity Gate", "viscosity_gate", 0.1, 5.0, 0.1, 1.5)
        
        Label(sidebar, text="─── Semantic Weights ───", bg='#222222', fg='#666666').pack(pady=(12,5))
        add_slider("Scene Understanding", "alpha_scene", 0.0, 1.0, 0.05, 0.35)
        add_slider("Boundary Context", "alpha_boundary", 0.0, 1.0, 0.05, 0.25)
        add_slider("Temporal Memory", "alpha_memory", 0.0, 1.0, 0.05, 0.20)
        add_slider("Prompt Influence", "alpha_prompt", 0.0, 1.0, 0.05, 0.20)
        add_slider("Steering Strength", "semantic_blend", 0.0, 1.0, 0.05, 0.6)
        
        # === PROMPT SECTION ===
        Label(sidebar, text="─── Prompt ───", bg='#222222', fg='#666666').pack(pady=(15,5))
        
        # Multi-line text box for prompt
        from tkinter import Text
        self.prompt_text = Text(sidebar, bg='#333333', fg='white', insertbackground='white',
                                height=4, wrap='word', font=("Arial", 10))
        self.prompt_text.insert('1.0', self.params["prompt"])
        self.prompt_text.pack(fill='x', pady=5)
        
        Button(sidebar, text="Update Prompt", command=self.update_prompt, 
               bg='#555555', fg='white', font=("Arial", 10)).pack(fill='x', pady=5)
        
        # === QUICK PROMPTS ===
        Label(sidebar, text="Quick Prompts:", bg='#222222', fg='#888888', anchor='w').pack(fill='x', pady=(10,2))
        
        quick_prompts = [
            ("🌊 Ocean", "deep ocean, bioluminescent creatures, underwater cave, ethereal light, 8k"),
            ("🌲 Forest", "enchanted forest, fractal trees, glowing mushrooms, mystical fog, detailed"),
            ("🏙️ Cyber", "cyberpunk city, neon lights, futuristic machinery, rain, blade runner"),
            ("🌌 Space", "cosmic nebula, stars, galaxies, space station, ethereal, 8k detailed"),
            ("🔮 Abstract", "abstract fractals, geometric patterns, vibrant colors, mathematical beauty"),
        ]
        
        for emoji_name, prompt in quick_prompts:
            btn = Button(sidebar, text=emoji_name, bg='#3a3a3a', fg='white',
                        command=lambda p=prompt: self.set_quick_prompt(p))
            btn.pack(fill='x', pady=1)
        
        # Spacer at bottom
        Label(sidebar, text="", bg='#222222', height=2).pack()
        
        # Video panel
        self.panel = Label(main_frame, bg='black')
        self.panel.pack(side='right', fill='both', expand=True)

    def start_model_loading(self):
        Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        try:
            # 1. Load Diffusion (Motor Brain)
            print("Loading Diffusion Model (Motor Brain)...")
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=torch.float16,
                safety_checker=None
            ).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.pipe.enable_attention_slicing()
            
            # 2. Load Semantic Steering (Vision Brain + Memory)
            print("Loading Semantic Steering Engine (Vision Brain)...")
            self.steering_engine = SemanticSteeringEngine(self.device)
            
            # 3. Create Embedding Injector (Bridge)
            self.embedding_injector = EmbeddingInjector(self.pipe, self.device)
            
            # Initialize frame
            self.last_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            
            # Warmup
            print("Warming up pipeline...")
            dummy_img = Image.fromarray(self.last_frame)
            with torch.inference_mode():
                self.pipe(prompt="warmup", image=dummy_img, strength=0.5,
                         num_inference_steps=4, guidance_scale=1.0)
            
            self.master.after(0, lambda: self.status_var.set("HIVEMIND Ready"))
            self.master.after(0, lambda: self.btn_toggle.config(
                state='normal', text="Start Hivemind Zoom", bg='#00aa00'))
            print("All systems loaded.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.master.after(0, lambda: self.status_var.set(f"Error: {e}"))

    def update_param(self, key, val):
        self.params[key] = val
    
    def update_prompt(self):
        # Get text from Text widget (line 1, char 0 to end, strip trailing newline)
        self.params["prompt"] = self.prompt_text.get('1.0', 'end-1c')
    
    def set_quick_prompt(self, prompt):
        """Set a quick prompt and update the text widget."""
        self.prompt_text.delete('1.0', 'end')
        self.prompt_text.insert('1.0', prompt)
        self.params["prompt"] = prompt
    
    def toggle_hivemind(self):
        self.use_hivemind = bool(self.hivemind_var.get())
    
    def update_mouse_steering(self, event):
        if self.panel.winfo_width() > 0:
            w = self.panel.winfo_width()
            h = self.panel.winfo_height()
            self.zoom_center_x = np.clip(event.x / w, 0, 1)
            self.zoom_center_y = np.clip(event.y / h, 0, 1)

    def apply_affine_zoom(self, image, scale, center_x, center_y):
        h, w = image.shape[:2]
        cx, cy = center_x * w, center_y * h
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def loop(self):
        while self.is_running:
            start_time = time.time()
            
            # 1. ZOOM
            with self.frame_lock:
                current_img = self.last_frame.copy()
            
            zoomed_img = self.apply_affine_zoom(
                current_img,
                self.params["zoom_speed"],
                self.zoom_center_x,
                self.zoom_center_y
            )
            
            # 2. VISCOSITY (Spatial Gate)
            structure_mask = self.viscosity_engine.get_viscosity_mask(
                zoomed_img,
                sensitivity=self.params["viscosity_gate"]
            )
            
            # 3. SEMANTIC STEERING (if enabled)
            pil_input = Image.fromarray(zoomed_img)
            
            with torch.inference_mode():
                if self.use_hivemind and self.steering_engine is not None:
                    # === HIVEMIND PATH ===
                    # Get steering embedding from vision brain
                    steering_emb = self.steering_engine.compute_steering_embedding(
                        zoomed_img,
                        structure_mask,
                        self.params["prompt"],
                        alpha_scene=self.params["alpha_scene"],
                        alpha_boundary=self.params["alpha_boundary"],
                        alpha_memory=self.params["alpha_memory"],
                        alpha_prompt=self.params["alpha_prompt"]
                    )
                    
                    # Create blended conditioning
                    prompt_embeds = self.embedding_injector.create_conditioning(
                        steering_emb,
                        self.params["prompt"],
                        blend_strength=self.params["semantic_blend"]
                    )
                    
                    # Generate with semantic steering
                    dream_output = self.pipe(
                        prompt_embeds=prompt_embeds,
                        image=pil_input,
                        strength=self.params["strength"],
                        num_inference_steps=4,
                        guidance_scale=1.0
                    ).images[0]
                    
                    stats_str = self.steering_engine.get_stats_string()
                else:
                    # === BASELINE PATH (no hivemind) ===
                    dream_output = self.pipe(
                        prompt=self.params["prompt"],
                        image=pil_input,
                        strength=self.params["strength"],
                        num_inference_steps=4,
                        guidance_scale=1.0
                    ).images[0]
                    stats_str = "Hivemind disabled"
            
            dream_np = np.array(dream_output)
            
            # 4. PHASE LOCK (Blend using viscosity mask)
            final_frame = (zoomed_img * structure_mask + 
                          dream_np * (1.0 - structure_mask)).astype(np.uint8)
            
            # Update feedback
            with self.frame_lock:
                self.last_frame = final_frame
            
            # Display
            self.update_display(final_frame)
            
            # Stats
            dt = time.time() - start_time
            fps = 1.0 / (dt + 1e-9)
            mode = "HIVEMIND" if self.use_hivemind else "BASELINE"
            self.master.after(0, lambda f=fps, m=mode: 
                            self.status_var.set(f"FPS: {f:.1f} | {m}"))
            self.master.after(0, lambda s=stats_str: self.stats_var.set(s))

    def update_display(self, frame_np):
        img = Image.fromarray(frame_np)
        imgtk = ImageTk.PhotoImage(image=img)
        
        def _update():
            self.panel.configure(image=imgtk)
            self.panel.image = imgtk
        
        self.master.after(0, _update)

    def toggle_zoom(self):
        if self.is_running:
            self.is_running = False
            self.btn_toggle.config(text="Start Hivemind Zoom", bg='#00aa00')
        else:
            self.is_running = True
            self.btn_toggle.config(text="Stop", bg='#aa0000')
            Thread(target=self.loop, daemon=True).start()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    INFINITE ZOOM HIVEMIND                                     ║
║                    Inter-Model Communication                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Vision Brain (CLIP)  ──────►  Memory Brain  ──────►  Motor Brain (Diffusion) ║
║      "What IS"                 "What WAS"              "Make it"              ║
║                                                                               ║
║  All brains share CLIP embedding space for lossless communication.           ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")
    
    root = Tk()
    app = InfiniteZoomHivemind(root)
    root.mainloop()