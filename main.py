import sys
from copy import deepcopy
import numpy as np
from tqdm import tqdm

import torch


def fit(model, loss, optimizer, train_dataloader, val_dataloader,
        num_epochs=100, patience=10, verbose=True, device=torch.device('cpu')):
    """Trains the model for a given number of epochs.

    Args:
        model: The PyTorch model to train.
        loss: The PyTorch loss function.
        optimizer: The PyTorch optimizer to use.
        train_dataloader: The training dataloader.
            Iterable of (input, target) pairs, both Tensors.
        val_dataloader: The validation dataloader.
            Iterable of (input, target) pairs, both Tensors.
        num_epochs: The number of epochs to train for.
        patience: The number of epochs to wait before early stopping.
        verbose: Whether to print progress.
        device: The device to use for training (e.g. 'cpu' or 'cuda').

    Returns:
        best_model: The best model.
        best_loss: The best loss.
        train_loss: The training loss.
        val_loss: The validation loss.
    """
    model.to(device)

    # Initialize the early stopping counter.
    steps_without_improvement = 0
    best_model = deepcopy(model)
    best_loss = np.inf

    # Initialize the lists for the training and validation loss.
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        with tqdm(train_dataloader, file=sys.stdout) as t:
            t.set_description(f'Epoch {epoch + 1}: Train: ')
            for x, y in t:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss_value = loss(y_pred, y[:, None])
                train_loss += loss_value.item()
                loss_value.backward()
                optimizer.step()
                t.set_postfix(loss=loss_value.detach().item())
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)

        # Validation loop
        with torch.no_grad():
            model.eval()
            val_loss = 0
            with tqdm(val_dataloader, file=sys.stdout) as t:
                t.set_description(f'Epoch {epoch + 1}: Val: ')
                for x, y in t:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    loss_value = loss(y_pred, y[:, None])
                    val_loss += loss_value.item()
                    t.set_postfix(loss=loss_value.detach().item())
                val_loss /= len(val_dataloader)
                val_losses.append(val_loss)

        if verbose:
            # Print progress
            string = f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}'
            if val_dataloader is not None:
                string += f' Val Loss: {val_loss:.4f}'
            print(string)

        # Check if the validation loss is improving (for early stopping).
        loss_value = val_loss if val_dataloader is not None else train_loss
        if loss_value < best_loss:
            best_loss = loss_value
            best_model = deepcopy(model)
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break
    return best_model, best_loss, train_losses, val_losses


@torch.no_grad()
def compute_accuracy(model, test_dataloader, device=torch.device('cpu')):
    r"""Computes the accuracy of the model on the test set.

    Args:
        model: The PyTorch model.
        test_dataloader: The test dataloader.
            Iterable of (input, target) pairs, both Tensors.
        device: The device to use for training (e.g. 'cpu' or 'cuda').

    Returns:
        accuracy: The mean accuracy of the model.
    """
    accuracy = 0
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        # sigmoid(y_pred) is the probability of the image being a dog.
        # If the probability is greater than 0.5 (y_pred > 0),
        # we predict that the image is a dog.
        # The accuracy is the mean of the correct predictions. (First, we sum
        # the sum over the batch, then we divide by the number of samples.)
        accuracy += torch.sum((y_pred[:, 0] > 0) == y).item()
    accuracy /= len(test_dataloader.dataset)
    return accuracy
