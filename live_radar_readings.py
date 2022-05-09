import time

import pyaudio
import wave
import timeit
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader

import timeit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class NeuralNet_1(nn.Module):
    def __init__(self):
        super(NeuralNet_1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.drop_out = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=26472, out_features=1000, bias=True)
        # self.fc2 = nn.Linear(in_features=1000, out_features=500, bias=True)
        self.fc3 = nn.Linear(in_features=1000, out_features=3, bias=True)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)

        out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc3(out)

        return out

def read_radar_data(i):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 1.38
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    #     print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        numpy_data = np.frombuffer(data, dtype=np.int16)
        frames.append(numpy_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    frames= np.array(frames).flatten()[:max_sig_len]
    plot_x = np.arange(len(frames))/RATE

    sig = torch.tensor(frames)
    sig_exp = torch.unsqueeze(torch.unsqueeze(sig, 0), 0).float()

    pred = model(sig_exp)
    print(torch.max(pred.data, 1)[1][0])

    plt.cla()
    plt.plot(plot_x, frames)
    plt.ylim(-5000, 5000)
    plt.tight_layout()

    # end = timeit.timeit()
    # print("this process took: ", (end-start), " s")

#     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()

max_sig_len = 105840
glob_min, glob_max = 0, 1000

model = NeuralNet_1()
model_path = 'Saved_models/NeuralNet_best.ckpt'
ex_data_path = 'data/test_data/example_sig'

with (open(ex_data_path + '.pkl', "rb")) as openfile:
  while True:
      try:
          ex_data = (pickle.load(openfile))
      except EOFError:
          break

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

ani = FuncAnimation(plt.gcf(), read_radar_data, interval=100)

plt.tight_layout()
plt.show()

