def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            correct += (pred.argmax(1) == y).sum().item()
            print(f"Batch {batch+1:>3d}/{num_batches:>3d} loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= int(size / 10)
    print(f"Train Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {train_loss:>8f} \n")


def test_loop(dataloader, model, loss_fn):
    import torch

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def validate_loop(dataloader, model, loss_fn):
    import torch

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validate_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            validate_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    validate_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100 * correct):>0.2f}, Avg loss: {validate_loss:>8f} \n")
