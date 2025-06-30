import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
# from quantum_cnn import ResNet20  # Commented out ResNet20
from adam_cnn import AdamOnlyCNN, AdamOnlyTrainer
from sklearn.metrics import f1_score, recall_score, confusion_matrix

def train_adam(num_epochs=30, batch_size=50, initial_lr=0.001, step_size=5, gamma=0.8):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # model = ResNet20(num_classes=10).to(device)  # Commented out ResNet20
    model = AdamOnlyCNN(num_classes=10).to(device)  # Use AdamOnlyCNN
    trainer = AdamOnlyTrainer(model, device)
    optimizer = trainer.optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(f"Initial Learning Rate: {optimizer.param_groups[0]['lr']}")
    print(f"Scheduler Type: step")

    train_losses, val_accuracies, lr_history = [], [], []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss = trainer.train_step(inputs, targets)
            epoch_loss += loss
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1}/{num_epochs} | Batch: {batch_idx}/{len(trainloader)} | Loss: {loss:.4f} | LR: {current_lr:.6f}')
        avg_train_loss = epoch_loss / len(trainloader)
        train_losses.append(avg_train_loss)
        # Validation accuracy only
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        val_acc = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
        val_accuracies.append(val_acc)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f'Epoch: {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}')

    # Compute metrics at the end
    f1 = f1_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    print(f'Final F1 Score: {f1:.4f}')
    print(f'Final Recall Score: {recall:.4f}')
    print(f'Final Confusion Matrix:\n{cm}')
    # Create results directories if they don't exist
    os.makedirs('results/npy', exist_ok=True)
    os.makedirs('results/images', exist_ok=True)
    np.save('results/npy/adam_f1_scores.npy', np.array([f1]))
    np.save('results/npy/adam_recall_scores.npy', np.array([recall]))
    np.save('results/npy/adam_confusion_matrices.npy', np.array([cm]))
    np.save('results/npy/adam_train_losses.npy', np.array(train_losses))
    np.save('results/npy/adam_val_accuracies.npy', np.array(val_accuracies))
    np.save('results/npy/adam_lr_history.npy', np.array(lr_history))
    # Plot results
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, val_accuracies, 'r-s', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Adam-Only Model Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(epochs, lr_history, 'g-^', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Step LR Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/images/adam_train_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved to 'results/images/adam_train_results.png'")

if __name__ == "__main__":
    train_adam() 