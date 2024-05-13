import torch
from torchvision import datasets, transforms #type: ignore
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm #type: ignore

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class MNISTConvNetTrainer():
    def __init__(self, model, train_loader, test_loader, lr=0.001, device='mps'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)

        self.model.to(self.device)
        print(f'Model created and moved to {device}')
        

    def train(self, num_epochs):
        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0
            correct = 0
            total = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Move data to device (GPU if available)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = epoch_loss / len(self.train_loader)
            epoch_acc = correct / total

            # compute test round
            test_loss, test_acc = self.test(self.test_loader)
            self.model.train()

            print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, Val Loss: {test_loss}, Val Acc: {test_acc}')
    

    def test(self, test_loader=None):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # compute also loss
                loss += self.criterion(outputs, labels)

        accuracy = correct / total
        loss = loss / len(self.test_loader)
        return loss, accuracy
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

# create MNIST dataloader
def get_mnist_dataloader(batch_size):
    # define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download and load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    # create a split for validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
