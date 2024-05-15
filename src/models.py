import torch
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader, RandomSampler, Subset
import torch.nn as nn
import torch.nn.functional as F


from typing import Tuple, Dict, List, Any
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed

# load config from../config.json
def load_config():
    import json
    with open("../mnist_config.json", "r") as f:
        config = json.load(f)
    return config


config = load_config()

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.1)  # Add dropout layer with dropout rate of 0.5

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to the output of the first fully connected layer
        x = self.fc2(x)
        return x


class MNISTConvNetTrainer():

    def __init__(self, model, train_loader, test_loader, lr=0.001, weight_decay=1e-6, device="mps"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.SGD(lr=lr, params=self.model.parameters(), weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)

        self.model.to(self.device)
        self.clients = {}
        print(f"Model created and moved to {device}")

    def train(self, num_epochs):
        self.model.train()

        for epoch in tqdm(range(num_epochs), ncols=80):
            epoch_loss = 0
            correct = 0
            total = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(
                    self.device
                )  # Move data to device (GPU if available)
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

            print(
                f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, Val Loss: {test_loss}, Val Acc: {test_acc}"
            )

    def test(self, test_loader=None):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
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
        print(f"Model saved to {path}")


class MNISTFederatedClient:

    def __init__(
            self, k: int, 
            local_epochs: int, 
            lr: float, 
            device="cpu", 
            dataset=None, 
            n_samples=1000, 
            weight_decay=1e-6, 
            criterion=nn.CrossEntropyLoss(),
            model=MNISTConvNet(),
            sample_method="iid"
        ) -> None:

        self.k = k
        self.lr = lr
        self.epochs = local_epochs
        self.device = torch.device(device)
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.criterion = criterion
        self.train_loader = dataset
        self.n_samples = n_samples  
        self.sample_method = sample_method
        # printu summary of all parameters
        print(f"> Client {k} created, lr: {lr}, epochs: {local_epochs}, samples: {n_samples}, device: {device}, weight_decay: {weight_decay}")

    def eval(self, dataset: DataLoader):
        correct = 0
        total = 0
        loss = 0
        # print batch size
        print(f"Batch size: {dataset.batch_size}")
        with torch.no_grad():
            for images, labels in dataset:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                loss += nn.CrossEntropyLoss()(outputs, labels)
                correct += (predicted == labels).sum().item()

        print(f"Local Acc: {correct/total}, Local Loss: {loss/len(dataset)}")

    def _sample_data_iid(self):
        subset = Subset(
            self.train_loader.dataset, 
            list(RandomSampler(self.train_loader.dataset, num_samples=self.n_samples)))
        return DataLoader(subset, batch_size=32, shuffle=True)
    
    def _sample_data_noniid(self):
        # select a a key from the mnist dataset randomly
        # random key
        key = torch.randint(0, 10, (1,)).item()
        # select all samples with that key
        subset = [s for s in self.train_loader.dataset if s[1] == self.k % 10]
        # sample n_samples from the subset
        subset = Subset(subset, list(RandomSampler(subset, num_samples=self.n_samples)))
        return DataLoader(subset, batch_size=32, shuffle=True)

    def __call__(self, state_dict ):

        sample_data = self._sample_data_iid() if self.sample_method == "iid" else self._sample_data_noniid()

        self.model.load_state_dict(state_dict)
        # model to device
        self.model.to(self.device)
        history = {
            "loss": [],
            "acc": []
        }
        for _ in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for images, labels in sample_data:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            epoch_loss = epoch_loss / len(sample_data)
            epoch_acc = correct / total
            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)
            #print(f"Local Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, client: {self.k}, total: {total}")

        return {
            "state_dict": self.model.state_dict(),
            "history": history
        }


class MNISTFederatedServer:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> None:

        self.test_loader = test_loader
        self.global_model = model
        self.criterion = nn.CrossEntropyLoss()
                

    def update_global_model(self, local_states: List[Dict]):
        self.global_model.load_state_dict(self.aggregate(local_states))
    

    def aggregate(self, local_states: List[Dict]):
        print(f"Aggregating {len(local_states)} local states\n")
        new_state = {}

        # initialize new state
        for key in local_states[0].keys():
            new_state[key] = torch.zeros_like(local_states[0][key])


        for state in local_states:
            for key in state.keys():
                new_state[key] += state[key]

        for key in new_state.keys():
            new_state[key] /= len(local_states) # assuming all clients have the same number of samples

        return new_state


    def start_train(self, clients, n_rounds, n_jobs=1):
        model_state = self.global_model.state_dict()   
        losses, accs = [], []
        for _ in range(n_rounds):
            
            local_states = Parallel(n_jobs=n_jobs)(
                delayed(client)(model_state) for client in tqdm(clients, ncols=80)
            )

            local_states = [state["state_dict"] for state in local_states]
            # create list of accuracies for eache model (a list of lists)

            self.update_global_model(local_states)

            test_loss, test_acc = self.test()
            losses.append(test_loss)
            accs.append(test_acc)
            print(f"Round {_}, Test Loss: {test_loss}, Test Acc: {test_acc}")
        return losses, accs
            

    def test(self):
        self.global_model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.global_model(images)
                loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.test_loader)
        self.global_model.train()
        return loss, accuracy


# create MNIST dataloader
def get_mnist_dataloader(batch_size):
    # define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # download and load the MNIST dataset
    train_dataset = datasets.MNIST(
        root=config["datapath"], train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root=config["datapath"], train=False, transform=transform
    )
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [50000, 10000]
    )

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def init_clients(num_clients, local_epochs, lr, train_dataset, n_samples=1000, device="cpu", weight_decay=1e-6, criterion=nn.CrossEntropyLoss(), model=MNISTConvNet(), sample_method="iid"):

    clients = []
    for i in range(num_clients):
        clients.append(
            MNISTFederatedClient(
                k=i, 
                local_epochs=local_epochs, 
                lr=lr, 
                dataset=train_dataset,
                n_samples=n_samples,
                device=device,
                weight_decay=weight_decay,
                sample_method=sample_method,
        )
    )
        
    return clients


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
