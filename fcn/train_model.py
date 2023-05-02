import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchinfo import summary
from model_utils import FCN

in_sz = 28 * 28

n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.manual_seed(random_seed)


model = FCN()
model = model.cuda()
model = model.eval()

x = torch.rand(1, in_sz).cuda()
summary(model, input_data=x)
y = model(x)
print("y=",y)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    torchvision.transforms.Lambda(torch.flatten)
])

train_ds = torchvision.datasets.MNIST('mnist', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)

test_ds = torchvision.datasets.MNIST('mnist', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size_test, shuffle=True)

batch_idx, (example_data, example_targets) = next(enumerate(test_loader))
print("example_data.shape:", example_data.shape)

batch_idx, (data, target) = next(enumerate(train_loader))
print("data.shape", data.shape)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

import os, shutil
if not os.path.isdir("results"):
    os.mkdir("results")
    print("results was created")


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to("cuda")
        target = target.to("cuda")
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), 'results/model_state.pt')
            torch.save(optimizer.state_dict(), 'results/optimizer_state.pt')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to("cuda")
            target = target.to("cuda")
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


n_epochs = 3
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

torch.save(model.state_dict(), 'results/model_state.pt')
torch.save(optimizer.state_dict(), 'results/optimizer_state.pt')
