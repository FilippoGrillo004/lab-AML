import torch
from torch import nn
import wandb  # <-- 1. IMPORTA WANDB
from models.custom_net import CustomNet
from eval import validate
from data.dataset import download_and_extract_tiny_imagenet, get_dataloaders

def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # INVIO AL DEVICE CORRETTO (AGNOSTICO)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    # Restituiamo i valori per poterli loggare
    return train_loss, train_accuracy

def main():
    # 2. INIZIALIZZA WANDB
    wandb.init(project="lab-AML", name="custom-net-run")

    download_and_extract_tiny_imagenet(extract_path='./tiny-imagenet')
    train_loader, val_loader = get_dataloaders(base_path='./tiny-imagenet/tiny-imagenet-200', batch_size=32)

    # 3. IMPOSTA IL DEVICE (Seleziona GPU se c'è, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0
    num_epochs = 10
    
    for epoch in range(1, num_epochs + 1):
        # Esegui train e salva i risultati
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        
        # Esegui validation (ora accetta il device e restituisce anche la loss)
        val_accuracy, val_loss = validate(model, val_loader, criterion, device)
        
        # 4. INVIA I DATI A WANDB PER I GRAFICI
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
        
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')
    
    # 5. CHIUDI WANDB A FINE TRAINING
    wandb.finish()

if __name__ == "__main__":
    main()