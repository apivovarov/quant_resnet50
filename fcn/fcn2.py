import torch
import torch.nn as nn
import torchvision
from model_utils import FCN2

in_sz = 28 * 28
batch_size_test = 8

random_seed = 1
torch.manual_seed(random_seed)

device = "cpu"
model = FCN2()
model = model.eval()


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])

test_ds = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

data, target = next(iter(test_loader))

print("Input:", data.shape, target.shape)

y=model(data)
print("Result:", y.shape)

tm = torch.jit.trace(model, data)
y=tm(data)
print("Result:", y.shape)

fname = "fcn2.pt"
torch.jit.save(tm, fname)
print(f"Saved {fname}")
