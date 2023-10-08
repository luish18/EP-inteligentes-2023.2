from ep_inteligentes_2023.io import leImagens
import torch
from torch import nn
from icecream import ic
from typing import Literal
import torchmetrics as tmetrics
from tqdm import tqdm


class dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type: Literal["Train", "Test"]):
        if dataset_type == "Test":
            paths = [
                "./data/Lung Segmentation Data/Lung Segmentation Data/Test/COVID-19/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Test/Non-COVID/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Test/Normal/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Val/COVID-19/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Val/Non-COVID/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Val/Normal/images/*.png",
            ]

            self.imgs, self.labels = leImagens(
                paths,
                classes=[0, 1, 2, 0, 1, 2],
            )

        else:
            paths = [
                "./data/Lung Segmentation Data/Lung Segmentation Data/Train/COVID-19/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Train/Non-COVID/images/*.png",
                "./data/Lung Segmentation Data/Lung Segmentation Data/Train/Normal/images/*.png",
            ]

            self.imgs, self.labels = leImagens(
                paths,
                classes=[0, 1, 2],
            )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return torch.tensor(self.imgs), torch.tensor(self.labels)


class Network(nn.Module):
    def __init__(self, k_size=(3, 3)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 64, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 32, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 16, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(16, 32, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(32, 64, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 128, kernel_size=k_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Flatten(),
            nn.Linear(128 * 224 * 224, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.model(x)


def train_loop(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer,
    metrics: tmetrics.MetricCollection | tmetrics.MetricTracker,
    device: torch.device,
):
    model.train()
    metrics.train()
    metrics.increment()
    running_loss = 0

    iterator = tqdm(dataloader, total=len(dataloader.dataset), leave=False)

    for features, labels in iterator:
        features.to(device)
        labels.to(device)

        pred = model(features)

        loss = loss_fn(pred, labels)
        running_loss += loss.item()

        metrics.update(torch.squeeze(pred), torch.squeeze(labels))
        update = metrics.compute()
        update.update({"train/loss": loss.item()})

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return running_loss / len(dataloader.dataset)


def test_loop(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    metrics: tmetrics.MetricCollection | tmetrics.MetricTracker,
    device: torch.device,
):
    model.eval()
    metrics.eval()
    metrics.increment()

    running_loss = 0
    iterator = tqdm(dataloader, leave=False)

    with torch.no_grad():
        for features, labels in iterator:
            pred = model(features)

            loss = loss_fn(pred, labels)
            running_loss += loss.item()

            metrics.update(torch.squeeze(pred), torch.squeeze(labels))
            update = metrics.compute()
            update.update({"test/loss": loss.item()})

    return running_loss / len(dataloader.dataset)

#|%%--%%| <W0oXdaWvA8|iiexgpmwrL>

device = torch.device("cpu")

training_data = dataset("Train")
test_data = dataset("Test")

BATCH_SIZE = 16
EPOCHS = 16
LR = 1e-3
#|%%--%%| <iiexgpmwrL|wqFFBUvlcN>

train_loader = torch.utils.data.DataLoader(
    dataset=training_data, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=True
)
#|%%--%%| <wqFFBUvlcN|ROd14kZ0Td>

NUM_LABELS = 3
test_metrics = tmetrics.MetricCollection(
    [
        tmetrics.classification.MulticlassAccuracy(num_classes=NUM_LABELS),
    ],
    prefix="test/",
)
train_metrics = tmetrics.MetricCollection(
    [
        tmetrics.classification.MulticlassAccuracy(num_classes=NUM_LABELS),
    ],
    prefix="train/",
)

model = Network()
model.to(device)

test_tracker = tmetrics.MetricTracker(test_metrics).to(device)
train_tracker = tmetrics.MetricTracker(train_metrics).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#|%%--%%| <ROd14kZ0Td|5WyE2whK5p>

iterator = tqdm(range(EPOCHS))
for t in iterator:
    train_loss = train_loop(
        dataloader=train_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics=train_tracker,
        device=device,
    )

    metrics = train_tracker.compute()
    update = {}

    for key in metrics.keys():
        update["epoch/" + key] = metrics[key]

    update.update({"epoch": t, "epoch/train/loss": train_loss})

    test_loss = test_loop(
        dataloader=test_loader,
        model=model,
        loss_fn=loss_fn,
        metrics=test_tracker,
        device=device,
    )

    metrics = test_tracker.compute()
    update = {}

    for key in metrics.keys():
        update["epoch/" + key] = metrics[key]
    update.update({"epoch": t, "epoch/test/loss": test_loss})
    iterator.set_postfix(update)

