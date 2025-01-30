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

FIELD_HEIGHT = 64   # each subregion's height
FIELD_WIDTH  = 256  # total width, subdivided into 4 x 64
SUBREGION_SIZE = 64 # each sub-block is 64 wide

# We'll define sub-blocks in the PDE:
# 1) Vision:   x from 0   .. 63
# 2) AudioIn:  x from 64  ..127
# 3) Brain:    x from 128 ..191
# 4) AudioOut: x from 192 ..255

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

AUDIO_SAMPLE_RATE = 44100
AUDIO_BLOCK_SIZE = 1024

# PDE parameters to tweak if everything is too uniform
INIT_KERNEL_STD = 0.3     # Convolution kernel init
INIT_FIELD_STD  = 0.02     # PDE field init
GLOBAL_COUPLING_MAX = 0.08 # upper clamp
DECAY_FACTOR = 0.997       # PDE field decay
PERTURB_SCALE = 0.001      # small external noise each step

# ============================================================
# 1. PDE + Gating System
# ============================================================

class MultimodalPDE(nn.Module):
    """
    A single PDE field of size [1, 1, 64, 256], subdivided into:
      - Vision (0..63),
      - AudioIn (64..127),
      - Brain (128..191),
      - AudioOut (192..255).
    We do partial gating with multiple PDE 'experts'.
    """

    def __init__(self, field_height=64, field_width=256,
                 num_experts=4, gating_hidden=16, device='cpu'):
        super().__init__()
        self.field_height = field_height
        self.field_width = field_width
        self.num_experts = num_experts
        self.device = device

        # The PDE field itself: shape [1, 1, H, W]
        self.field = nn.Parameter(
            torch.zeros(1, 1, field_height, field_width, device=self.device),
            requires_grad=False
        )

        # PDE experts: small 3x3 conv filters
        self.experts = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            for _ in range(num_experts)
        ])
        for expert in self.experts:
            nn.init.normal_(expert.weight, mean=0.0, std=INIT_KERNEL_STD)

        # Gating net: output distribution over num_experts+1 => last is "no update"
        self.gating_net = nn.Sequential(
            nn.Conv2d(1, gating_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(gating_hidden, num_experts+1, kernel_size=3, padding=1)
        )

        # PDE update parameters
        self.decay = DECAY_FACTOR
        self.global_coupling = 0.015
        self.critical_variance = 0.04
        self.critical_adjust_rate = 0.00005

        self.to(self.device)

        # random init of the field
        self.field.data = INIT_FIELD_STD * torch.randn_like(self.field)

    def forward(self):
        """
        One PDE update step.
        """
        # gating
        gating_logits = self.gating_net(self.field)  # shape [1, num_experts+1, 64, 256]
        gating_dist = F.softmax(gating_logits, dim=1)  # same shape

        # PDE updates from each expert
        expert_updates = []
        for exp in self.experts:
            upd = exp(self.field)  # shape [1,1,64,256]
            expert_updates.append(upd)

        # stack => [1, num_experts, 1, 64, 256]
        stack_updates = torch.stack(expert_updates, dim=1)

        # separate gating
        gating_experts = gating_dist[:, :self.num_experts, :, :]  # [1, num_experts, 64, 256]
        gating_no_update = gating_dist[:, self.num_experts:self.num_experts+1, :, :]

        # Weighted sum
        gating_experts_5d = gating_experts.unsqueeze(2)
        weighted_updates = gating_experts_5d * stack_updates
        combined_update = weighted_updates.sum(dim=1)  # => [1,1,64,256]

        # partial gating => multiply by (1 - gating_no_update)
        partial_pde = combined_update * (1.0 - gating_no_update)

        # PDE field update
        new_field = self.field + self.global_coupling * partial_pde

        # decay + clamp
        new_field = self.decay * new_field
        new_field = torch.tanh(new_field)

        self.field.data = new_field

        # tiny random noise to keep it from going uniform
        noise = (torch.rand_like(self.field) - 0.5) * PERTURB_SCALE
        self.field.data = self.field + noise

        # self-organized critical check
        self._adjust_coupling()

        return self.field

    def _adjust_coupling(self):
        # measure amplitude variance, adjust self.global_coupling
        arr = self.field.detach().cpu().numpy()
        var = np.var(arr)
        diff = var - self.critical_variance
        self.global_coupling -= self.critical_adjust_rate * diff
        self.global_coupling = max(0.0, min(GLOBAL_COUPLING_MAX, self.global_coupling))

    # ---------------------------------------------------------
    # Sub-block read/write methods
    # ---------------------------------------------------------

    def get_subblock(self, region_idx):
        """
        region_idx: 0=Vision,1=AudioIn,2=Brain,3=AudioOut
        returns a (64x64) slice as a torch tensor [1,1,64,64]
        """
        x_start = region_idx * SUBREGION_SIZE
        x_end   = x_start + SUBREGION_SIZE
        return self.field[..., :, x_start:x_end]

    def set_subblock(self, region_idx, data_64x64):
        """
        data_64x64 shape: [1,1,64,64]
        Overwrite PDE field in that region with 'data_64x64'
        """
        x_start = region_idx * SUBREGION_SIZE
        x_end   = x_start + SUBREGION_SIZE
        self.field.data[..., :, x_start:x_end] = data_64x64

# ============================================================
# 3. The "MultimodalBrain" system
# ============================================================

class MultimodalBrain:
    """
    Coordinates the PDE, audio in/out, camera feed, and merges them in a single loop.
    We'll have:
      - PDE sub-block 0 (Vision):    we store a 64x64 grayscale from camera each step
      - PDE sub-block 1 (AudioIn):  we store some audio input representation
      - PDE sub-block 2 (Brain):    PDE manipulates it
      - PDE sub-block 3 (AudioOut): we read it, convert to audio, output
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

        # We'll keep ring buffers or minimal queues for audio
        self.audio_in_queue = queue.deque(maxlen=5)
        self.audio_out_queue = queue.deque(maxlen=5)

        # Setup microphone + speaker streams
        self.stream = None

        # Setup camera
        self.cam = None
        self.running = False

    # ---------------------------------------------------------
    # 3A. Audio callbacks
    # ---------------------------------------------------------

    def audio_callback(self, indata, outdata, frames, time_info, status):
        # read mic input
        mono_in = indata[:, 0] if indata.ndim > 1 else indata
        # store in queue
        self.audio_in_queue.append(mono_in.copy())

        # produce output
        if self.audio_out_queue:
            out_chunk = self.audio_out_queue.popleft()
            outdata[:] = out_chunk.reshape(-1,1)
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
            self.stream = None

    # ---------------------------------------------------------
    # 3B. Camera
    # ---------------------------------------------------------

    def start_camera(self, cam_index=0):
        self.cam = cv2.VideoCapture(cam_index)
        if not self.cam.isOpened():
            print(f"Warning: Could not open camera index={cam_index}")

    def stop_camera(self):
        if self.cam:
            self.cam.release()
            self.cam = None

    def read_camera_grayscale_64x64(self):
        if not self.cam:
            return None
        ret, frame = self.cam.read()
        if not ret:
            return None
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize to 64x64
        small = cv2.resize(gray, (64,64))
        # normalize
        arr = small.astype(np.float32) / 255.0
        # shape => [1,1,64,64]
        arr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        return arr_t

    # ---------------------------------------------------------
    # 3C. Inserting Audio Input into PDE & generating audio out
    # ---------------------------------------------------------

    def audio_in_to_pde(self, audio_chunk):
        """
        audio_chunk shape ~ (1024,)
        We'll map it into a 64x64 block (4096 samples).
        """
        needed = FIELD_HEIGHT * SUBREGION_SIZE  # 64*64=4096
        audio_chunk = audio_chunk[:needed] if len(audio_chunk) >= needed \
                      else np.pad(audio_chunk, (0, needed - len(audio_chunk)), mode='constant')
        # reshape to 64x64
        arr = audio_chunk.reshape(FIELD_HEIGHT, SUBREGION_SIZE)
        # normalize [-1..1]
        arr = np.clip(arr, -1.0, 1.0).astype(np.float32)

        arr_t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        # insert into PDE sub-block 1
        self.pde.set_subblock(1, arr_t)

    def pde_to_audio_out(self):
        """
        Read PDE sub-block 3 (64x64), flatten to 4096,
        downsample by 4 => 1024 samples
        """
        data_64 = self.pde.get_subblock(3)
        arr = data_64.squeeze().detach().cpu().numpy()  # shape (64,64)
        flattened = arr.flatten()  # 4096
        chunk_1024 = flattened.reshape(-1,4).mean(axis=1)  # shape (1024,)
        chunk_1024 = np.clip(chunk_1024, -1, 1).astype(np.float32)
        return chunk_1024

    # ---------------------------------------------------------
    # 3D. The main PDE loop
    # ---------------------------------------------------------

    def step(self):
        """
        One iteration: read camera, store in PDE subregion 0
                       read audio_in queue, store in PDE subregion 1
                       PDE forward step
                       read PDE subregion 3 => audio_out queue
        """
        # Vision
        cam_data = self.read_camera_grayscale_64x64()
        if cam_data is not None:
            self.pde.set_subblock(0, cam_data)

        # Audio in
        if self.audio_in_queue:
            chunk = self.audio_in_queue.popleft()
            self.audio_in_to_pde(chunk)

        # PDE update
        self.pde()

        # produce audio out
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
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.im = None
        self.running = True

    def init_plot(self):
        field = self.brain.pde.field.detach().cpu().numpy()[0,0,...]
        self.im = self.ax.imshow(field, cmap='plasma', vmin=-1, vmax=1)
        self.ax.set_title("Multimodal PDE Brain")
        return [self.im]

    def update_plot(self, frame):
        # do PDE step
        self.brain.step()
        # update image
        field = self.brain.pde.field.detach().cpu().numpy()[0,0,...]
        self.im.set_data(field)
        return [self.im]

    def run(self):
        anim = FuncAnimation(
            self.fig, self.update_plot, frames=None,
            init_func=self.init_plot, interval=50, blit=True
        )
        plt.show()
        # once user closes window
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
    root.title("Multimodal PDE Setup")

    # Query all audio devices
    devices = sd.query_devices()

    # Separate lists for input and output devices
    input_devices = []
    output_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev['name']))
        if dev['max_output_channels'] > 0:
            output_devices.append((i, dev['name']))

    # Create tkinter variables
    selected_input = tk.StringVar(value="None")
    selected_output = tk.StringVar(value="None")
    selected_camera = tk.StringVar(value="0")

    # Dropdown for Input Device
    label_in = tk.Label(root, text="Select Input Device:")
    label_in.pack(pady=(10,0))
    combo_in = ttk.Combobox(root, textvariable=selected_input, state="readonly",
                            values=["None"] + [f"{d[0]}: {d[1]}" for d in input_devices])
    combo_in.current(0)  # default "None"
    combo_in.pack()

    # Dropdown for Output Device
    label_out = tk.Label(root, text="Select Output Device:")
    label_out.pack(pady=(10,0))
    combo_out = ttk.Combobox(root, textvariable=selected_output, state="readonly",
                             values=["None"] + [f"{d[0]}: {d[1]}" for d in output_devices])
    combo_out.current(0)  # default "None"
    combo_out.pack()

    # Dropdown for Camera Index
    label_cam = tk.Label(root, text="Camera Index:")
    label_cam.pack(pady=(10,0))
    # We guess there may be up to 5 cameras
    combo_cam = ttk.Combobox(root, textvariable=selected_camera, state="readonly",
                             values=[str(i) for i in range(5)])
    combo_cam.current(0)
    combo_cam.pack()

    def on_start():
        # Parse audio input device index
        input_choice = combo_in.get()
        if input_choice == "None":
            input_index = None
        else:
            input_index = int(input_choice.split(":")[0])

        # Parse audio output device index
        output_choice = combo_out.get()
        if output_choice == "None":
            output_index = None
        else:
            output_index = int(output_choice.split(":")[0])

        # Parse camera index
        cam_index = int(combo_cam.get())

        # Close GUI
        root.destroy()

        # Now launch the PDE code with these settings
        app = MultimodalDemo(camera_index=cam_index,
                             audio_in=input_index,
                             audio_out=output_index)
        app.run()

    button_start = tk.Button(root, text="Start", command=on_start)
    button_start.pack(pady=(20,10))

    root.mainloop()

# ============================================================
# 6. Main script entry point
# ============================================================

def main():
    launch_gui()

if __name__ == "__main__":
    main()
