import torch
from torch import nn
from models.custom_net import CustomNet
from eval import validate
from data.dataset import download_and_extract_tiny_imagenet, get_dataloaders

def train_one_epoch(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

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

def main():
    # 1. Assicurati che i dati ci siano (scarica se serve)
    download_and_extract_tiny_imagenet(extract_path='./tiny-imagenet')
    
    # 2. Carica i dati (la cartella val si auto-sistemerà grazie alla tua funzione)
    train_loader, val_loader = get_dataloaders(base_path='./tiny-imagenet/tiny-imagenet-200', batch_size=32)

    # 2. Inizializza Modello, Loss e Ottimizzatore
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 3. Loop di Addestramento
    best_acc = 0
    num_epochs = 10
    
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer)
        
        # Alla fine di ogni epoca, fai validazione
        val_accuracy = validate(model, val_loader, criterion)
        
        # Salva la best accuracy
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    main()