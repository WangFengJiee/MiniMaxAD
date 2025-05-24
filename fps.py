import torch
from torch import nn
import time
import numpy as np
from models.unireplknet import unireplknet_n
from models.de_unireplknet import de_unireplknet_n


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder, self.bn = unireplknet_n()
        self.decoder = de_unireplknet_n()
        self.encoder.reparameterize_unireplknet()
        self.bn.reparameterize_unireplknet()
        self.decoder.reparameterize_unireplknet()

    def forward(self, x):
        en = self.encoder(x)
        de = self.decoder(self.bn(en))
        return en, de


model = Model().cuda().eval()
batch_size = 16
input_size = (3, 256, 256)

with torch.no_grad():
    for _ in range(50):
        x = torch.randn((batch_size, *input_size)).cuda()
        _ = model(x)
    torch.cuda.synchronize()

t_all = []
for i in range(500):
    x = torch.randn((batch_size, *input_size)).cuda()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    with torch.no_grad():
        _ = model(x)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    t_all.append(t2 - t1)

times = np.array(t_all)
mean_time = np.mean(times)
std_time = np.std(times)

trimmed = np.sort(times)[int(len(times) * 0.05):int(len(times) * 0.95)]
trimmed_mean = np.mean(trimmed)

print(f'Average time: {mean_time:.6f}s ± {std_time:.6f}s')
print(f'Trimmed average time (5%-95%): {trimmed_mean:.6f}s')
print(f'Average FPS: {batch_size / mean_time:.2f}')
print(f'Fastest time: {np.min(times):.6f}s | FPS: {batch_size / np.min(times):.2f}')
print(f'Slowest time: {np.max(times):.6f}s | FPS: {batch_size / np.max(times):.2f}')
