import torch
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader, RandomSampler, Subset
import torch.nn as nn
import torch.nn.functional as F


from typing import Tuple, Dict, List, Any
from tqdm import tqdm  # type: ignore
from joblib import Parallel, delayed
from utils import load_config

to_cpu = lambda x: x.cpu()
config = load_config()


class ConvNet(nn.Module):
    """
    Convolutional Neural Network model for image classification.

    Args:
        None

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with input channels=1 and output channels=8.
        conv2 (nn.Conv2d): Second convolutional layer with input channels=8 and output channels=16.
        conv3 (nn.Conv2d): Third convolutional layer with input channels=16 and output channels=32.
        fc1 (nn.Linear): First fully connected layer with input features=32*3*3 and output features=64.
        fc2 (nn.Linear): Second fully connected layer with input features=64 and output features=10.
        dropout (nn.Dropout): Dropout layer with dropout rate of 0.1.

    Methods:
        forward(x): Performs forward pass through the network.

    Returns:
        x (torch.Tensor): Output tensor after passing through the network.
    """

    def __init__(self):
        super(ConvNet, self).__init__()
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
        x = self.dropout(
            x
        )  # Apply dropout to the output of the first fully connected layer
        x = self.fc2(x)
        return x


class ConvNetTrainer:
    """
    A class for training and testing a convolutional neural network model.

    Args:
        model (torch.nn.Module): The convolutional neural network model.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the testing dataset.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 1e-6.
        device (str, optional): The device to use for training and testing. Defaults to "mps".
        optimizer (str, optional): The optimizer to use. Supported options are "SGD", "Adam", and "AdamW". Defaults to "SGD".

    Attributes:
        model (torch.nn.Module): The convolutional neural network model.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        test_loader (torch.utils.data.DataLoader): The data loader for the testing dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        criterion (torch.nn.Module): The loss function used for training and testing.
        device (torch.device): The device used for training and testing.
        tr_losses (list): List to store the training losses.
        ts_losses (list): List to store the testing losses.
        tr_accs (list): List to store the training accuracies.
        ts_accs (list): List to store the testing accuracies.

    Methods:
        train(num_epochs, earlystopping=False): Trains the model for the specified number of epochs.
        test(test_loader=None): Tests the model on the testing dataset.
        save_model(path): Saves the trained model to the specified path.
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        lr=0.001,
        weight_decay=1e-6,
        device="mps",
        optimizer="SGD",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

        if optimizer == "SGD":
            print("Using SGD")
            self.optimizer = torch.optim.SGD(
                lr=lr, params=self.model.parameters(), weight_decay=weight_decay
            )
        elif optimizer == "Adam":
            print("Using Adam")
            self.optimizer = torch.optim.Adam(
                lr=lr, params=self.model.parameters(), weight_decay=weight_decay
            )
        elif optimizer == "AdamW":
            print("Using AdamW")
            self.optimizer = torch.optim.AdamW(
                lr=lr, params=self.model.parameters(), weight_decay=weight_decay
            )
        else:
            raise ValueError("Optimizer not supported")

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device(device)

        self.model.to(self.device)
        self.tr_losses = []
        self.ts_losses = []
        self.tr_accs = []
        self.ts_accs = []
        print(f"Model created and moved to {device}")

    def train(self, num_epochs, earlystopping=False):
        """
        Trains the model for the specified number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.
            earlystopping (bool, optional): Whether to use early stopping. Defaults to False.
        """
        self.model.train()
        best_weights = self.model.state_dict()

        print(
            f"Total number of trainable parameters: {
            sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }"
        )

        progress_bar = tqdm(range(num_epochs), ncols=128)
        for epoch in progress_bar:

            epoch_loss, _ = train_loop(
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                sample_data=self.train_loader,
                device=self.device,
            )

            # compute test round
            test_loss, test_acc = self.test(self.test_loader)
            self.tr_losses.append(epoch_loss)
            self.ts_losses.append(test_loss)

            if earlystopping:
                if early_stopping(val_loss=self.ts_losses, patience=5):
                    self.model.load_state_dict(best_weights)
                    break
            else:
                best_weights = self.model.state_dict()

            self.model.train()

            progress_bar.set_postfix_str(
                # setting postfix string with 3 decimal points
                f"Loss: {epoch_loss:.3f}, VL: {test_loss:.3f}, VA: {test_acc*100:.2f}"
            )

    def test(self, test_loader=None):
        """
        Tests the model on the testing dataset.

        Args:
            test_loader (torch.utils.data.DataLoader, optional): The data loader for the testing dataset.
                If not provided, the default test loader will be used.

        Returns:
            tuple: A tuple containing the testing loss and accuracy.
        """
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
        """
        Saves the trained model to the specified path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")


class FederatedClient:
    """
    Represents a federated learning client.

    Args:
        k (int): The client identifier.
        local_epochs (int): The number of local epochs to train the model.
        lr (float): The learning rate for the optimizer.
        device (str, optional): The device to use for training. Defaults to "mps".
        dataset (torch.utils.data.Dataset, optional): The training dataset. Defaults to None.
        n_samples (int, optional): The number of samples to use for training. Defaults to 1000.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to 1e-6.
        criterion (torch.nn.Module, optional): The loss criterion. Defaults to nn.CrossEntropyLoss().
        model (torch.nn.Module, optional): The model architecture. Defaults to None.
        sample_method (str, optional): The method for sampling the data. Defaults to "iid".

    Attributes:
        k (int): The client identifier.
        lr (float): The learning rate for the optimizer.
        epochs (int): The number of local epochs to train the model.
        device (torch.device): The device to use for training.
        model (torch.nn.Module): The model architecture.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.Module): The loss criterion.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        n_samples (int): The number of samples to use for training.
        sample_method (str): The method for sampling the data.

    Methods:
        eval(dataset): Evaluates the model on a given dataset.
        _sample_data_iid(): Samples data in an i.i.d. manner.
        _sample_data_noniid(): Samples data in a non-i.i.d. manner.
        __call__(state_dict): Trains the model using federated learning.

    """

    def __init__(
        self,
        k: int,
        local_epochs: int,
        lr: float,
        device="mps",
        dataset=None,
        n_samples=1000,
        weight_decay=1e-6,
        criterion=nn.CrossEntropyLoss(),
        model=None,
        sample_method="iid",
    ) -> None:
        """
        Initializes a FederatedClient object.

        Args:
            k (int): The client identifier.
            local_epochs (int): The number of local epochs to train the model.
            lr (float): The learning rate for the optimizer.
            device (str, optional): The device to use for training. Defaults to "mps".
            dataset (torch.utils.data.Dataset, optional): The training dataset. Defaults to None.
            n_samples (int, optional): The number of samples to use for training. Defaults to 1000.
            weight_decay (float, optional): The weight decay for the optimizer. Defaults to 1e-6.
            criterion (torch.nn.Module, optional): The loss criterion. Defaults to nn.CrossEntropyLoss().
            model (torch.nn.Module, optional): The model architecture. Defaults to None.
            sample_method (str, optional): The method for sampling the data. Defaults to "iid".
        """
        self.k = k
        self.lr = lr
        self.epochs = local_epochs
        self.device = torch.device(device)
        self.model = model
        if model is None:
            self.model = ConvNet()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )
        self.criterion = criterion
        self.train_loader = dataset
        self.n_samples = n_samples
        self.sample_method = sample_method
        # print summary of all parameters
        print(
            f"> Client {k} created, lr: {lr}, epochs: {local_epochs}, samples: {n_samples}, device: {device}, weight_decay: {weight_decay}"
        )

    def eval(self, dataset: DataLoader):
        """
        Evaluates the model on a given dataset.

        Args:
            dataset (torch.utils.data.DataLoader): The dataset to evaluate the model on.
        """
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
        """
        Samples data in an i.i.d. manner.

        Returns:
            torch.utils.data.DataLoader: The data loader for the sampled data.
        """
        subset = Subset(
            self.train_loader.dataset,
            list(RandomSampler(self.train_loader.dataset, num_samples=self.n_samples)),
        )
        return DataLoader(subset, batch_size=32, shuffle=True)

    def _sample_data_noniid(self):
        """
        Samples data in a non-i.i.d. manner.

        Returns:
            torch.utils.data.DataLoader: The data loader for the sampled data.
        """
        # select a key from the mnist dataset randomly
        subset_a = [s for s in self.train_loader.dataset if s[1] == self.k % 10]
        subset_b = [s for s in self.train_loader.dataset if s[1] == (self.k + 1) % 10]
        # sample n_samples from the subset
        subset_a = Subset(
            subset_a, list(RandomSampler(subset_a, num_samples=self.n_samples // 2))
        )
        subset_b = Subset(
            subset_b, list(RandomSampler(subset_b, num_samples=self.n_samples // 2))
        )

        subset = torch.utils.data.ConcatDataset([subset_a, subset_b])

        return DataLoader(subset, batch_size=32, shuffle=True)

    def __call__(self, state_dict):
        """
        Trains the model using federated learning.

        Args:
            state_dict (dict): The state dictionary of the model.

        Returns:
            dict: The state dictionary of the trained model and the training history.
        """
        sample_data = (
            self._sample_data_iid()
            if self.sample_method == "iid"
            else self._sample_data_noniid()
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        history = {"loss": [], "acc": []}
        for _ in range(self.epochs):

            epoch_loss, epoch_acc = train_loop(
                model=self.model,
                criterion=self.criterion,
                optimizer=self.optimizer,
                sample_data=sample_data,
                device=self.device,
            )

            history["loss"].append(epoch_loss)
            history["acc"].append(epoch_acc)
            # print(f"Local Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, client: {self.k}, total: {total}")

        return {"state_dict": self.model.state_dict(), "history": history}


class FederatedServer:
    """
    A class representing a federated server for distributed model training.

    Args:
        model (nn.Module): The global model to be trained.
        test_loader (DataLoader): The data loader for the test dataset.
        device (str, optional): The device to be used for training. Defaults to "cpu".

    Attributes:
        test_loader (DataLoader): The data loader for the test dataset.
        global_model (nn.Module): The global model to be trained.
        device (str): The device used for training.
        criterion (nn.Module): The loss function for training.

    Methods:
        _update_global_model: Updates the global model by aggregating the local model states.
        _aggregate: Aggregates the local model states.
        start_train: Starts the federated training process.
        test: Evaluates the global model on the test dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cpu",
    ) -> None:
        """
        Initializes a new instance of the FederatedServer class.

        Args:
            model (nn.Module): The global model to be trained.
            test_loader (DataLoader): The data loader for the test dataset.
            device (str, optional): The device to be used for training. Defaults to "cpu".
        """
        self.test_loader = test_loader
        self.global_model = model
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.global_model.to(self.device)

    def _update_global_model(self, local_states: List[Dict]):
        """
        Updates the global model by aggregating the local model states.

        Args:
            local_states (List[Dict]): A list of dictionaries representing the local model states.
        """
        self.global_model.load_state_dict(self._aggregate(local_states))

    def _aggregate(self, local_states: List[Dict]):
        """
        Aggregates the local model states.

        Args:
            local_states (List[Dict]): A list of dictionaries representing the local model states.

        Returns:
            Dict: The aggregated model state.
        """
        new_state = {}

        # initialize new state
        for key in local_states[0].keys():
            new_state[key] = torch.zeros_like(local_states[0][key])

        for state in local_states:
            for key in state.keys():
                new_state[key] += state[key]

        for key in new_state.keys():
            new_state[key] = new_state[key] / len(
                local_states
            )  # assuming all clients have the same number of samples

        return new_state

    def start_train(self, clients, n_rounds, n_jobs=1):
        """
        Starts the federated training process.

        Args:
            clients: The list of clients participating in the training.
            n_rounds (int): The number of training rounds.
            n_jobs (int, optional): The number of parallel jobs to run. Defaults to 1.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing the lists of losses and accuracies.
        """
        model_state = self.global_model.state_dict()
        losses, accs = [], []
        pbar = tqdm(range(n_rounds), ncols=80)

        for _ in pbar:
            local_states = Parallel(n_jobs=n_jobs)(
                delayed(c)(model_state) for c in clients
            )

            local_states = [state["state_dict"] for state in local_states]

            self._update_global_model(local_states)

            test_loss, test_acc = self.test()
            losses.append(test_loss)
            accs.append(test_acc)
            pbar.set_postfix_str(f"Loss: {test_loss:.3f}, Acc: {test_acc*100:.2f}")

        losses = list(map(to_cpu, losses))
        return losses, accs

    def test(self):
        """
        Evaluates the global model on the test dataset.

        Returns:
            Tuple[float, float]: A tuple containing the test loss and accuracy.
        """
        self.global_model.eval()
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss += self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.test_loader)
        self.global_model.train()
        return loss, accuracy


def init_clients(
    num_clients,
    local_epochs,
    lr,
    train_dataset,
    n_samples=1000,
    device="cpu",
    weight_decay=1e-6,
    criterion=nn.CrossEntropyLoss(),
    model=ConvNet(),
    sample_method="iid",
):
    """
    Initializes a list of federated clients.

    Args:
        num_clients (int): The number of clients to initialize.
        local_epochs (int): The number of local epochs for each client.
        lr (float): The learning rate for each client.
        train_dataset (torch.utils.data.Dataset): The training dataset.
        n_samples (int, optional): The number of samples to use for each client. Defaults to 1000.
        device (str, optional): The device to use for training. Defaults to "cpu".
        weight_decay (float, optional): The weight decay for each client. Defaults to 1e-6.
        criterion (torch.nn.Module, optional): The loss criterion for each client. Defaults to nn.CrossEntropyLoss().
        model (torch.nn.Module, optional): The model architecture for each client. Defaults to ConvNet().
        sample_method (str, optional): The method for sampling data for each client. Defaults to "iid".

    Returns:
        list: A list of initialized federated clients.
    """
    clients = []
    for i in range(num_clients):
        clients.append(
            FederatedClient(
                k=i,
                local_epochs=local_epochs,
                lr=lr,
                dataset=train_dataset,
                n_samples=n_samples,
                device=device,
                weight_decay=weight_decay,
                sample_method=sample_method,
                model=model,
            )
        )

    return clients


def early_stopping(val_loss, patience=5):
    if len(val_loss) > patience:
        if all(val_loss[-1] > val_loss[-(i + 1)] for i in range(patience)):
            return True
    return False


def train_loop(model, optimizer, criterion, sample_data, device):
    """
    Trains the given model using the provided optimizer and criterion on the given sample data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        criterion (torch.nn.Module): The loss function used to compute the training loss.
        sample_data (torch.utils.data.DataLoader): The data loader containing the training samples.
        device (torch.device): The device on which the model and data should be loaded.

    Returns:
        tuple: A tuple containing the average training loss and the training accuracy for the epoch.
    """
    epoch_loss = 0
    correct = 0
    total = 0

    model.train()
    for images, labels in sample_data:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = epoch_loss / len(sample_data)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
