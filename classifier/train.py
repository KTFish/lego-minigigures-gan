import torch
from tqdm import tqdm
import config


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str = config.DEVICE,
):
    """Performs one training step.

    Args:
        model (troch.nn.Module): Model on which the step will be performed.
        dataloader (torch.utils.data.DataLoader): Train dataloader.
        optimizer (torch.optim.Optimizer): Oprimizer.
        loss_fn (torch.nn.Module): Loss Function.
        device (str, optional): Device on which the model will be stored. A GPU is desirable. Defaults to device.

    Returns:
        train_loss, train_acc (float, float): Train Loss and Train Accuracy scores.
    """

    model.to(device)
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device, dtype=torch.float32)
        y_logits = model(X).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        # print(y_logits.shape, y.shape)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        # y_pred_classes = torch.argmax(y_pred, dim=1)
        acc = torch.eq(y_pred, y).sum().item() / y.size(0)
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str = config.DEVICE,
):
    model.eval()
    model.to(device)
    with torch.inference_mode():
        test_loss, test_acc = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device, dtype=torch.float32)
            y_logits = model(X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))

            loss = loss_fn(y_logits, y)
            test_loss += loss.item()

            # y_pred_classes = torch.argmax(y_pred, dim=1)
            acc = torch.eq(y_pred, y).sum().item() / len(y)
            test_acc += acc

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    num_epochs: int = config.EPOCHS,
    device: str = config.DEVICE,
):
    """Trains the model using train_step and test_step helper functions.

    Args:
        model (troch.nn.Module): _description_
        train_dataloader (torch.utils.data.DataLoader): _description_
        test_dataloader (torch.utils.data.DataLoader): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss_fn (torch.nn.Module): _description_
        num_epochs (int, optional): _description_. Defaults to 1.
        device (str, optional): _description_. Defaults to device.
    """
    results = {"train_acc": [], "train_loss": [], "test_acc": [], "test_loss": []}

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        print(
            f"Epoch: {epoch + 1} | train loss: {train_loss:.3f} | train acc: {train_acc:.3f} | test loss: {test_loss:.3f} | test acc: {test_acc:.3f}"
        )
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)
        results["test_acc"].append(test_acc)
        results["test_loss"].append(test_loss)
    torch.save(model.state_dict(), "my_model.pth")
    return results
