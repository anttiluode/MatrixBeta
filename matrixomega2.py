"""
MATRIX OMEGA v1.1: THE SELF-NARRATING HIVEMIND
==============================================
New Feature: "RESET NARRATIVE" button to clear hallucinations.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from transformers import CLIPModel, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageTk
from tkinter import Tk, Label, Scale, HORIZONTAL, Frame, Button, StringVar, Text, Checkbutton, IntVar, Canvas, Scrollbar
from threading import Thread, Lock
import time
import os
from collections import deque

# Disable HF Hub transfer for stability
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# =============================================================================
# 1. THE NARRATOR (BLIP - System 2)
# =============================================================================
class NarratorBrain:
    def __init__(self, device):
        self.device = device
        print("Loading Narrator (BLIP)...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()
        print("Narrator Ready.")

    def describe(self, image_np):
        h, w = image_np.shape[:2]
        crop = image_np[h//4:h*3//4, w//4:w*3//4]
        
        pil_img = Image.fromarray(crop)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device, torch.float16)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20, min_new_tokens=5)
            
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

# =============================================================================
# 2. THE VISCOSITY ENGINE (Spatial Gate)
# =============================================================================
class FractalViscosityMap:
    def __init__(self, device):
        self.device = device
        
    def get_viscosity_mask(self, image_np, sensitivity=1.0):
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image_np.astype(np.float32) / 255.0
            
        sigma = 3
        mean = cv2.GaussianBlur(gray, (0, 0), sigma)
        mean_sq = cv2.GaussianBlur(gray**2, (0, 0), sigma)
        variance = mean_sq - mean**2
        
        std_dev = np.sqrt(np.maximum(0, variance))
        viscosity = np.clip((std_dev - 0.05) * sensitivity * 15.0, 0.0, 1.0)
        viscosity = cv2.GaussianBlur(viscosity, (0, 0), 2.0)
        
        return np.repeat(viscosity[:, :, np.newaxis], 3, axis=2)

# =============================================================================
# 3. THE STEERING ENGINE (CLIP - System 1)
# =============================================================================
class SemanticSteeringEngine:
    def __init__(self, device):
        self.device = device
        print("Loading Steering Engine (CLIP)...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        self.memory = deque(maxlen=8) 
        
    def get_steering_vector(self, image_np, prompt):
        pil_img = Image.fromarray(image_np)
        img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        txt_inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            img_emb = self.model.get_image_features(**img_inputs)
            txt_emb = self.model.get_text_features(**txt_inputs)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
            
        self.memory.append(img_emb)
        mem_emb = torch.cat(list(self.memory)).mean(dim=0, keepdim=True)
        mem_emb /= mem_emb.norm(dim=-1, keepdim=True)
        
        steering = 0.4 * img_emb + 0.3 * mem_emb + 0.3 * txt_emb
        steering /= steering.norm(dim=-1, keepdim=True)
        return steering

# =============================================================================
# 4. THE INJECTOR (Bridge to Diffusion)
# =============================================================================
class EmbeddingInjector:
    def __init__(self, pipe, device):
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.device = device
        
    def create_conditioning(self, steering_emb, prompt, blend=0.6):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_emb = self.text_encoder(tokens.input_ids.to(self.device))[0]
        steering_expanded = steering_emb.unsqueeze(1).expand(-1, 77, -1).to(text_emb.dtype)
        final_emb = (1 - blend) * text_emb + blend * steering_expanded
        return final_emb

# =============================================================================
# 5. MATRIX OMEGA (The App)
# =============================================================================
class MatrixOmega:
    def __init__(self, master):
        self.master = master
        self.master.title("MATRIX OMEGA v1.1 - The Self-Narrating Hivemind")
        self.master.geometry("1200x850")
        self.master.configure(bg='#050505')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.frame_lock = Lock()
        self.running = False
        self.last_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        
        self.zoom_x, self.zoom_y = 0.5, 0.5
        self.anchor_prompt = "bioluminescent forest, fractal roots, detailed, 8k"
        self.current_narrative = ""
        
        self.params = {
            "zoom": 1.05,
            "strength": 0.35,
            "viscosity": 1.5,
            "steering": 0.5,
            "narrator_influence": 0.5,
            "narrator_interval": 15
        }
        
        self.setup_ui()
        Thread(target=self.load_systems, daemon=True).start()

    def setup_ui(self):
        left = Frame(self.master, bg='#111', width=300)
        left.pack(side='left', fill='y')
        
        self.panel = Label(self.master, bg='black')
        self.panel.pack(side='right', fill='both', expand=True)
        self.panel.bind('<Motion>', self.mouse_move)
        
        Label(left, text="MATRIX OMEGA", font=("Consolas", 16, "bold"), bg='#111', fg='#0f0').pack(pady=10)
        
        self.status = StringVar(value="Booting Systems...")
        Label(left, textvariable=self.status, bg='#111', fg='white', font=("Arial", 10)).pack()
        
        self.btn = Button(left, text="START ENGINE", command=self.toggle, bg='#333', fg='white', state='disabled')
        self.btn.pack(fill='x', padx=10, pady=10)
        
        # Narrative Display
        Label(left, text="--- THE NARRATIVE ---", bg='#111', fg='#888').pack(pady=5)
        self.lbl_narrative = Label(left, text="...", bg='#000', fg='#0ff', wraplength=280, justify='left', height=3)
        self.lbl_narrative.pack(padx=10, fill='x')
        
        # RESET BUTTON
        Button(left, text="RESET NARRATIVE (Clear Hallucination)", command=self.reset_narrative, 
               bg='#500', fg='white').pack(fill='x', padx=10, pady=5)
        
        # Sliders
        def slider(lbl, key, minv, maxv, defv):
            Label(left, text=lbl, bg='#111', fg='gray').pack(pady=(10,0))
            s = Scale(left, from_=minv, to=maxv, resolution=0.01, orient=HORIZONTAL, bg='#111', fg='white', highlightthickness=0)
            s.set(defv)
            s.pack(fill='x', padx=10)
            s.config(command=lambda v: self.update_param(key, float(v)))
            
        slider("Zoom Speed", "zoom", 1.01, 1.20, 1.05)
        slider("Dream Strength", "strength", 0.1, 0.9, 0.35)
        slider("Viscosity Gate", "viscosity", 0.0, 5.0, 1.5)
        slider("Visual Steering (CLIP)", "steering", 0.0, 1.0, 0.5)
        slider("Narrator Influence", "narrator_influence", 0.0, 1.0, 0.5)
        
        self.use_narrator = IntVar(value=1)
        Checkbutton(left, text="Enable Narrator (Auto-Prompt)", variable=self.use_narrator, bg='#111', fg='white', selectcolor='#333').pack(pady=10)
        
        Label(left, text="Anchor Prompt:", bg='#111', fg='gray').pack(pady=(10,0))
        self.txt_prompt = Text(left, height=4, bg='#222', fg='white', insertbackground='white')
        self.txt_prompt.insert('1.0', self.anchor_prompt)
        self.txt_prompt.pack(padx=10, pady=5, fill='x')
        Button(left, text="Update Anchor", command=self.update_prompt, bg='#444', fg='white').pack(fill='x', padx=10)

    def load_systems(self):
        try:
            print("Loading Diffusion...")
            self.pipe = AutoPipelineForImage2Image.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=torch.float16, safety_checker=None).to(self.device)
            self.pipe.set_progress_bar_config(disable=True)
            self.viscosity = FractalViscosityMap(self.device)
            self.clip = SemanticSteeringEngine(self.device)
            self.narrator = NarratorBrain(self.device)
            self.injector = EmbeddingInjector(self.pipe, self.device)
            
            print("Warming up...")
            self.last_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(self.last_frame)
            self.pipe(prompt="warmup", image=img, num_inference_steps=1).images[0]
            
            self.status.set("SYSTEMS ONLINE")
            self.btn.config(state='normal', bg='#060')
        except Exception as e:
            self.status.set(f"Error: {e}")
            print(e)

    def update_param(self, key, val):
        self.params[key] = val
        
    def update_prompt(self):
        self.anchor_prompt = self.txt_prompt.get("1.0", "end-1c")
        
    def reset_narrative(self):
        self.current_narrative = ""
        self.lbl_narrative.config(text="[Narrative Reset]")
        print("Narrative Wiped. Returning to Anchor.")

    def mouse_move(self, event):
        w, h = self.panel.winfo_width(), self.panel.winfo_height()
        if w > 0 and h > 0:
            self.zoom_x = event.x / w
            self.zoom_y = event.y / h

    def toggle(self):
        if self.running:
            self.running = False
            self.btn.config(text="START ENGINE", bg='#060')
        else:
            self.running = True
            self.btn.config(text="STOP", bg='#600')
            Thread(target=self.loop, daemon=True).start()

    def loop(self):
        frame_count = 0
        while self.running:
            start = time.time()
            
            h, w = self.last_frame.shape[:2]
            cx, cy = int(self.zoom_x * w), int(self.zoom_y * h)
            M = cv2.getRotationMatrix2D((cx, cy), 0, self.params['zoom'])
            zoomed = cv2.warpAffine(self.last_frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            
            mask = self.viscosity.get_viscosity_mask(zoomed, sensitivity=self.params['viscosity'])
            
            active_prompt = self.anchor_prompt
            
            if self.use_narrator.get() and frame_count % self.params['narrator_interval'] == 0:
                caption = self.narrator.describe(zoomed)
                self.current_narrative = caption
                self.master.after(0, lambda c=caption: self.lbl_narrative.config(text=f'"{c}"'))
            
            if self.current_narrative:
                active_prompt = f"{self.anchor_prompt}, {self.current_narrative}"
            
            steering_vec = self.clip.get_steering_vector(zoomed, active_prompt)
            cond = self.injector.create_conditioning(steering_vec, active_prompt, blend=self.params['steering'])
            
            pil_zoom = Image.fromarray(zoomed)
            with torch.inference_mode():
                out = self.pipe(
                    prompt_embeds=cond,
                    image=pil_zoom,
                    strength=self.params['strength'],
                    num_inference_steps=4,
                    guidance_scale=1.0
                ).images[0]
            
            dream_np = np.array(out)
            final = (zoomed * mask + dream_np * (1 - mask)).astype(np.uint8)
            
            self.last_frame = final
            frame_count += 1
            
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(final))
            self.master.after(0, lambda i=imgtk: self.update_panel(i))
            
            dt = time.time() - start
            if dt < 0.05: time.sleep(0.05 - dt)

    def update_panel(self, img):
        self.panel.configure(image=img)
        self.panel.image = img

if __name__ == "__main__":
    root = Tk()
    app = MatrixOmega(root)
    root.mainloop()