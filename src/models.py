import torch
from torchvision import datasets, transforms #type: ignore
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


from typing import Tuple, Dict, List, Any
from tqdm import tqdm #type: ignore

# load config from../config.json
def load_config():
    import json
    with open('../config.json', 'r') as f:
        config = json.load(f)
    return config

config = load_config()

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
        self.clients = {}
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


class MNISTFederatedClient():
    def __init__(self, k: int, local_epochs:int, lr:float, device='cpu') -> None:
        self.k = k
        self.lr = lr
        self.epochs = local_epochs
        self.device = torch.device(device)

    def eval(self, dataset: DataLoader, model):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataset:
                images,labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
            
            
    
    def __call__(self, dataset: DataLoader, model_state: Dict):
        # training with federated learning
        model = MNISTConvNet()
        model.load_state_dict(model_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(self.epochs):
            for images, labels in dataset:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                # only updating local weights
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        # return local model state
        # evaluate the model
        self.eval(dataset, model)
        return model.state_dict()


class MNISTFederatedServer():
    def __init__(
            self, t: int, 
            lr: float, 
            model: nn.Module, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            criterion: nn.Module = nn.CrossEntropyLoss(),
            C=1,
        ) -> None:
    
        self.train_loader = train_loader # remove this, each client should have its own data
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.k = 0
        self.model = model
        self.lr = lr
        self.clients: Dict[int, MNISTFederatedClient] = {}
        self.rounds = t
        self.C = C
        self.m = max([self.C*self.k, 1])
        self.criterion = criterion


    def init_clients(self, num_clients, local_epochs, lr):
        self.k = num_clients
        for i in range(num_clients):
            self.clients[i] = MNISTFederatedClient(
                k=1, 
                local_epochs=local_epochs, 
                lr=lr
            )


    def update_global_model(self, local_states: List[Dict]):
        # aggregate local model states
        # ratio = [len_indexes[i] / sum(len_indexes) for i in range(self.k
        new_states_dict = {}
        for param in self.model.state_dict().keys():
            new_states = sum([state[param] for state in local_states]) 
            new_states_dict[param] = new_states / self.k
        
        # update global model
        self.model.load_state_dict(new_states_dict)


    def _copy_state(self, state):
        new_state = {}
        for key, value in state.items():
            new_state[key] = value.clone()
        return new_state
    
    def start_train(self):

        client_data = []

        for _ in range(self.rounds):
            local_states = []
            len_indexes  = []
            indexes = torch.randperm(len(self.train_loader.dataset))
            for i in range(self.k):
                start = i * self.m
                end = min((i+1)*self.m, len(indexes))
                len_indexes.append(end-start)
                loader = DataLoader(self.train_loader.dataset, batch_size=self.m, sampler=indexes[start:end])
                client_data.append(loader)
        
            for client, loader in zip(self.clients.values(), client_data):
                # create a copy of self.model.state_dict() for each client
                state_copy = self._copy_state(self.model.state_dict())
                s = client(loader, self.model.state_dict())
                local_states.append(s)
            
            self.update_global_model(local_states)

            # evaluate the model
            self.model.eval()
            test_loss, test_acc = self.test(self.test_loader)
            print(f'Round {_}, Test Loss: {test_loss}, Test Acc: {test_acc}')

    def test(self, test_loader=None):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.test_loader)
        return loss, accuracy
    


# create MNIST dataloader
def get_mnist_dataloader(batch_size):
    # define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # download and load the MNIST dataset
    train_dataset = datasets.MNIST(root=config['datapath'], train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=config['datapath'], train=False, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')
