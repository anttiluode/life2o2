import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sounddevice as sd
import cv2
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import threading
import queue
import time
import random

import tkinter as tk
from tkinter import ttk

# ============================================================
# Configurable parameters
# ============================================================

FIELD_HEIGHT = 64   # PDE subregion height
FIELD_WIDTH  = 256  # PDE total width (4 sub-blocks, each 64 wide)
SUBREGION_SIZE = 64 # size of each sub-block's width

# Sub-block layout in PDE:
#  0: Vision   (x=0..63)
#  1: AudioIn  (x=64..127)
#  2: Brain    (x=128..191)
#  3: AudioOut (x=192..255)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE  = 1024

# PDE parameters
INIT_KERNEL_STD   = 0.3       # PDE conv filter init
INIT_FIELD_STD    = 0.02      # PDE field init scale
GLOBAL_COUPLING_MAX = 0.08    # clamp
DECAY_FACTOR = 0.997
PERTURB_SCALE = 0.001
NEAR_CRITICAL_TARGET_VARIANCE = 0.04
CRITICAL_ADJUST_RATE = 0.00005

# ============== Spiking Neuron (LIF) Parameters =============
NUM_NEURONS_BRAIN = 20  # how many LIF neurons in the Brain sub-block
NEURON_RADIUS     = 3   # PDE injection radius
PDE_TO_NEURON_RAD = 3   # PDE amplitude sampling radius
VTHRESH           = 1.0
RESET_POT         = 0.0
MEM_LEAK          = 0.9  # simple leak factor
NEURON_TO_PDE_STRENGTH = 0.02
PDE_TO_NEURON_STRENGTH = 0.02

# ============================================================
# 1. PDE + Gating System
# ============================================================

class MultimodalPDE(nn.Module):
    """
    PDE field of shape [1, 1, 64, 256], subdivided into:
      Vision   (x=0..63),
      AudioIn  (x=64..127),
      Brain    (x=128..191),
      AudioOut (x=192..255).

    We do partial gating with multiple PDE 'experts'.
    """

    def __init__(self,
                 field_height=64,
                 field_width=256,
                 num_experts=4,
                 gating_hidden=16,
                 device='cpu'):
        super().__init__()
        self.field_height = field_height
        self.field_width  = field_width
        self.num_experts  = num_experts
        self.device       = device

        # PDE field param => [1,1, H, W]
        self.field = nn.Parameter(
            torch.zeros(1,1,field_height,field_width,device=self.device),
            requires_grad=False
        )

        # PDE experts => small 3x3 conv filters
        self.experts = nn.ModuleList([
            nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
            for _ in range(num_experts)
        ])
        for expert in self.experts:
            nn.init.normal_(expert.weight, mean=0.0, std=INIT_KERNEL_STD)

        # Gating net => output (num_experts+1) => last channel is "no update"
        self.gating_net = nn.Sequential(
            nn.Conv2d(1, gating_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(gating_hidden, num_experts+1, kernel_size=3, padding=1)
        )

        # PDE update config
        self.decay     = DECAY_FACTOR
        self.global_coupling = 0.015
        self.target_variance = NEAR_CRITICAL_TARGET_VARIANCE

        # random init
        self.field.data = INIT_FIELD_STD * torch.randn_like(self.field)

        self.to(self.device)

    def forward(self):
        """
        Single PDE update step:
          - partial gating => pick PDE expert per cell
          - near-critical coupling => measure variance, adjust
          - small noise
        """
        gating_logits = self.gating_net(self.field)  # [1, (experts+1), H, W]
        gating_dist = F.softmax(gating_logits, dim=1)
        gating_for_experts = gating_dist[:, :self.num_experts, :, :]   # [1,num_experts,H,W]
        gating_no_update   = gating_dist[:, self.num_experts:, :, :]   # [1,1,H,W]

        # PDE from each expert
        expert_outs = []
        for exp in self.experts:
            out = exp(self.field)  # [1,1,H,W]
            expert_outs.append(out)
        stack_outs = torch.stack(expert_outs, dim=1)  # => [1, num_experts, 1,H,W]

        # Weighted sum => shape [1,1,H,W]
        gating_5d = gating_for_experts.unsqueeze(2)   # => [1, num_experts, 1,H,W]
        combined_update = (stack_outs * gating_5d).sum(dim=1)

        # partial gating => (1 - gating_no_update)
        partial_pde = combined_update * (1.0 - gating_no_update)

        # PDE field update
        new_field = self.field + self.global_coupling * partial_pde
        # decay + clamp
        new_field = self.decay * new_field
        new_field = torch.tanh(new_field)

        self.field.data = new_field

        # small random noise
        noise = (torch.rand_like(self.field)-0.5) * PERTURB_SCALE
        self.field.data = self.field + noise

        self._adjust_coupling()

    def _adjust_coupling(self):
        # measure variance, push toward target
        arr = self.field.detach().cpu().numpy()
        var = np.var(arr)
        diff= var - self.target_variance
        self.global_coupling -= CRITICAL_ADJUST_RATE * diff
        self.global_coupling = max(0.0, min(GLOBAL_COUPLING_MAX, self.global_coupling))

    # Sub-block read/write
    def get_subblock(self, region_idx):
        """
        region_idx: 0=Vision,1=AudioIn,2=Brain,3=AudioOut
        Returns shape [1,1,64,64]
        """
        x_start = region_idx * SUBREGION_SIZE
        x_end   = x_start + SUBREGION_SIZE
        return self.field[..., :, x_start:x_end]

    def set_subblock(self, region_idx, data_64x64):
        x_start = region_idx * SUBREGION_SIZE
        x_end   = x_start + SUBREGION_SIZE
        self.field.data[..., :, x_start:x_end] = data_64x64

    # Optional: a quick function to inject amplitude at a certain (x,y) with radius
    def inject_amplitude(self, x, y, radius, amount):
        """
        Add 'amount' to PDE cells in a circle around (x,y). Coordinates in PDE sub-block space.
        PDE shape => [1,1,64,256]. x in [0..255], y in [0..63].
        """
        x1 = max(0, x-radius)
        x2 = min(self.field_width, x+radius+1)
        y1 = max(0, y-radius)
        y2 = min(self.field_height,y+radius+1)

        grid_x = torch.arange(x1,x2,device=self.field.device).view(-1,1).expand(-1,y2-y1)
        grid_y = torch.arange(y1,y2,device=self.field.device).view(1,-1).expand(x2-x1,-1)
        dist_sq= (grid_x - x)**2 + (grid_y - y)**2
        mask= (dist_sq <= radius**2)

        subfield= self.field[..., grid_y, grid_x]
        subfield[0,0,mask] += amount

    def sample_amplitude(self, x, y, radius):
        """
        Return average PDE amplitude in circle around (x,y) of 'radius'.
        """
        x1 = max(0, x-radius)
        x2 = min(self.field_width, x+radius+1)
        y1 = max(0, y-radius)
        y2 = min(self.field_height,y+radius+1)

        grid_x = torch.arange(x1,x2,device=self.field.device).view(-1,1).expand(-1,y2-y1)
        grid_y = torch.arange(y1,y2,device=self.field.device).view(1,-1).expand(x2-x1,-1)
        dist_sq= (grid_x - x)**2 + (grid_y - y)**2
        mask= (dist_sq <= radius**2)

        subfield= self.field[..., grid_y, grid_x]
        val= subfield[0,0,mask].mean()
        return val.item()

# ============================================================
# 2. LIF Neuron for Brain sub-block
# ============================================================

class LIFNeuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.V = 0.0
        self.spiked = False

    def update(self, input_current):
        # Simple leak integrator
        self.V = MEM_LEAK * self.V + input_current
        if self.V >= VTHRESH:
            self.spiked = True
            self.V = RESET_POT
        else:
            self.spiked = False

# ============================================================
# 3. The "MultimodalBrain" system (with spiking neurons)
# ============================================================

class MultimodalBrain:
    """
    PDE sub-blocks:
     0 => Vision,   1 => AudioIn,
     2 => Brain,    3 => AudioOut

    We'll place discrete LIF neurons in sub-block #2 (Brain).
    Each PDE step:
       - If any neuron spiked last step => inject PDE amplitude
       - PDE update
       - PDE amplitude => updates each neuron's membrane
    Audio & camera logic remain same: sub-block0 gets camera, sub-block1 gets audio in,
    sub-block3 => read out to speaker.
    """
    def __init__(self):
        self.device = DEVICE
        self.pde = MultimodalPDE(
            field_height=FIELD_HEIGHT,
            field_width=FIELD_WIDTH,
            num_experts=4,
            gating_hidden=16,
            device=self.device
        )
        self.stream = None
        self.cam    = None

        # Audio in/out queues
        self.audio_in_queue  = queue.deque(maxlen=5)
        self.audio_out_queue = queue.deque(maxlen=5)

        # Create spiking neurons in Brain sub-block => x in [128..191]
        self.neurons = []
        for _ in range(NUM_NEURONS_BRAIN):
            x = random.randint(128, 191)
            y = random.randint(0, 63)
            self.neurons.append(LIFNeuron(x,y))

    # ------------- Audio -------------
    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(status)
        mono_in = indata[:,0] if indata.ndim>1 else indata
        self.audio_in_queue.append(mono_in.copy())

        if self.audio_out_queue:
            chunk = self.audio_out_queue.popleft()
            outdata[:] = chunk.reshape(-1,1)
        else:
            outdata.fill(0)

    def start_audio(self, input_device=None, output_device=None):
        sd.default.samplerate = AUDIO_SAMPLE_RATE
        sd.default.blocksize  = AUDIO_BLOCK_SIZE
        if input_device is not None:
            sd.default.device[0] = input_device
        if output_device is not None:
            sd.default.device[1] = output_device

        self.stream = sd.Stream(
            channels=1,
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=AUDIO_BLOCK_SIZE,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

    def stop_audio(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream=None

    # ------------- Camera -------------
    def start_camera(self, cam_index=0):
        self.cam = cv2.VideoCapture(cam_index)
        if not self.cam.isOpened():
            print(f"Warning: Could not open camera index={cam_index}")

    def stop_camera(self):
        if self.cam:
            self.cam.release()
            self.cam=None

    def read_camera_grayscale_64x64(self):
        if not self.cam:
            return None
        ret, frame = self.cam.read()
        if not ret:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small= cv2.resize(gray,(64,64))
        arr  = small.astype(np.float32)/255.0
        arr_t= torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        return arr_t

    # ------------- PDE + Audio Sub-block -------------
    def audio_in_to_pde(self, audio_chunk):
        """
        audio_chunk => 1024 samples => reshape to [64x64]=4096 if enough
        clamp to [-1..1], store in PDE sub-block #1 => shape [1,1,64,64]
        """
        needed = FIELD_HEIGHT*SUBREGION_SIZE  # 64x64=4096
        arr = audio_chunk[:needed]
        if len(arr)<needed:
            arr = np.pad(arr,(0, needed-len(arr)))

        arr_2d = arr.reshape(64,64)
        arr_2d = np.clip(arr_2d,-1,1).astype(np.float32)
        arr_t  = torch.from_numpy(arr_2d).unsqueeze(0).unsqueeze(0).to(self.device)
        self.pde.set_subblock(1, arr_t)

    def pde_to_audio_out(self):
        """
        read PDE sub-block#3 => shape(64,64)=4096
        flatten => downsample by factor4 => 1024
        """
        data_64 = self.pde.get_subblock(3)
        arr     = data_64.squeeze().detach().cpu().numpy()
        flattened= arr.flatten()
        chunk_1024 = flattened.reshape(-1,4).mean(axis=1)
        chunk_1024 = np.clip(chunk_1024,-1,1).astype(np.float32)
        return chunk_1024

    # ------------- Spiking Neuron Coupling -------------
    def spiking_neurons_step(self):
        """
        1) If neuron spiked last step => inject PDE amplitude
        2) PDE update
        3) PDE => neuron membrane
        """
        # (1) Neuron -> PDE
        for nrn in self.neurons:
            if nrn.spiked:
                # inject amplitude in PDE
                self.pde.inject_amplitude(
                    x=nrn.x,
                    y=nrn.y,
                    radius=NEURON_RADIUS,
                    amount=NEURON_TO_PDE_STRENGTH
                )

        # (2) PDE update
        self.pde()

        # (3) PDE -> Neuron
        for nrn in self.neurons:
            local_amp = self.pde.sample_amplitude(
                x=nrn.x,
                y=nrn.y,
                radius=PDE_TO_NEURON_RAD
            )
            input_current = PDE_TO_NEURON_STRENGTH*local_amp
            nrn.update(input_current)

    # ------------- Main step -------------
    def step(self):
        # Vision sub-block #0
        cam_data = self.read_camera_grayscale_64x64()
        if cam_data is not None:
            self.pde.set_subblock(0, cam_data)

        # AudioIn sub-block #1
        if self.audio_in_queue:
            chunk = self.audio_in_queue.popleft()
            self.audio_in_to_pde(chunk)

        # PDE + spiking neuron synergy
        self.spiking_neurons_step()

        # AudioOut sub-block #3
        out_chunk = self.pde_to_audio_out()
        self.audio_out_queue.append(out_chunk)

# ============================================================
# 4. PDE Runner with Matplotlib
# ============================================================

class MultimodalDemo:
    def __init__(self, camera_index=0, audio_in=None, audio_out=None):
        """
        :param camera_index: which webcam index to open (int)
        :param audio_in: input device index or None
        :param audio_out: output device index or None
        """
        self.brain = MultimodalBrain()
        self.brain.start_camera(camera_index)
        self.brain.start_audio(input_device=audio_in, output_device=audio_out)

        # Setup matplot
        self.fig, self.ax = plt.subplots(figsize=(8,4))
        self.im = None
        self.running = True

    def init_plot(self):
        field = self.brain.pde.field.detach().cpu().numpy()[0,0,...]
        self.im = self.ax.imshow(field, cmap='plasma', vmin=-1, vmax=1)
        self.ax.set_title("Multimodal PDE Brain + Spiking Neurons")

        # Plot neuron positions as white dots
        for nrn in self.brain.neurons:
            self.ax.plot(nrn.x, nrn.y, 'wo', markersize=3)

        return [self.im]

    def update_plot(self, frame):
        # Step PDE + spiking neurons + camera + audio
        self.brain.step()

        # update display
        field = self.brain.pde.field.detach().cpu().numpy()[0,0,...]
        self.im.set_data(field)
        return [self.im]

    def run(self):
        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            frames=None,
            init_func=self.init_plot,
            interval=50,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
        self.shutdown()

    def shutdown(self):
        self.running = False
        self.brain.stop_camera()
        self.brain.stop_audio()

# ============================================================
# 5. GUI portion
# ============================================================

def launch_gui():
    """
    A simple Tkinter GUI to pick audio input, audio output, and webcam index,
    then launch the PDE animation once the user clicks 'Start'.
    """
    root = tk.Tk()
    root.title("Multimodal PDE Setup (Spiking)")

    import sounddevice as sd
    devices = sd.query_devices()

    input_devices = []
    output_devices= []
    for i, dev in enumerate(devices):
        if dev['max_input_channels']>0:
            input_devices.append((i, dev['name']))
        if dev['max_output_channels']>0:
            output_devices.append((i, dev['name']))

    selected_input = tk.StringVar(value="None")
    selected_output= tk.StringVar(value="None")
    selected_camera= tk.StringVar(value="0")

    label_in = tk.Label(root,text="Select Input Device:")
    label_in.pack(pady=(10,0))
    combo_in = ttk.Combobox(
        root,
        textvariable=selected_input,
        state="readonly",
        values=["None"]+[f"{d[0]}: {d[1]}" for d in input_devices]
    )
    combo_in.current(0)
    combo_in.pack()

    label_out= tk.Label(root,text="Select Output Device:")
    label_out.pack(pady=(10,0))
    combo_out= ttk.Combobox(
        root,
        textvariable=selected_output,
        state="readonly",
        values=["None"]+[f"{d[0]}: {d[1]}" for d in output_devices]
    )
    combo_out.current(0)
    combo_out.pack()

    label_cam= tk.Label(root,text="Camera Index:")
    label_cam.pack(pady=(10,0))
    combo_cam= ttk.Combobox(
        root,
        textvariable=selected_camera,
        state="readonly",
        values=[str(i) for i in range(5)]
    )
    combo_cam.current(0)
    combo_cam.pack()

    def on_start():
        # parse
        in_choice= combo_in.get()
        if in_choice=="None":
            in_dev=None
        else:
            in_dev= int(in_choice.split(":")[0])

        out_choice= combo_out.get()
        if out_choice=="None":
            out_dev=None
        else:
            out_dev= int(out_choice.split(":")[0])

        cam_idx= int(combo_cam.get())

        root.destroy()
        app= MultimodalDemo(camera_index=cam_idx, audio_in=in_dev, audio_out=out_dev)
        app.run()

    button_start= tk.Button(root, text="Start", command=on_start)
    button_start.pack(pady=(20,10))

    root.mainloop()

# ============================================================
# 6. Main script entry point
# ============================================================

def main():
    launch_gui()

if __name__ == "__main__":
    main()
