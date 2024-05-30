from models import ConvNet, FederatedServer, init_clients
from utils import get_mnist_dataloader

import argparse, json

train, val, test = get_mnist_dataloader(64)
model = ConvNet()
server = FederatedServer(model=model, test_loader=val)

# add arguments
parser = argparse.ArgumentParser(description="FedMNIST")

parser.add_argument("--n_clients", type=int, default=16, help="number of clients to initialize")
parser.add_argument( "--local_epochs", type=int, default=10, help="number of local epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument( "--n_samples", type=int, default=512, help="number of samples per client")
parser.add_argument("--device", type=str, default="mps", help="device to train on")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay for optimizer")
parser.add_argument("--iid",type=bool,default=True, help="whether to sample clients iid or not")
parser.add_argument("--rounds", type=int, default=64, help="number of rounds to train for")
parser.add_argument("--jobs", type=int, default=8, help="number of jobs to run in parallel")

args = parser.parse_args()

# just printing the config
print(json.dumps(vars(args), indent=4))

input("Press Enter to start training...")


if __name__ == "__main__":
    clients = init_clients(
        num_clients=args.n_clients,
        local_epochs=args.local_epochs,
        lr=args.lr,
        train_dataset=train,
        n_samples=args.n_samples,
        device=args.device,
        weight_decay=args.weight_decay,
        sample_method="iid" if args.iid else "non-iid",
    )
    server.start_train(clients, n_rounds=args.rounds, n_jobs=args.jobs)

    server.global_model.eval()
    server.test_loader = test
    server.test()
