import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from quantum_cnn import QuantumEnhancedResNet20
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import os

from typing import Tuple, Optional, Literal

class AdaptiveQuantumTrainer:
    def __init__(self, 
                 model: QuantumEnhancedResNet20, 
                 device: torch.device,
                 initial_lr: float = 0.001,
                 scheduler_params: dict = None,
                 quantum_strategy: str = 'adaptive'):
        self.model = model
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
        self.quantum_strategy = quantum_strategy
        # Only use step scheduler
        if scheduler_params is None:
            scheduler_params = {'step_size': 2, 'gamma': 0.5}
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=scheduler_params.get('step_size', 2), 
            gamma=scheduler_params.get('gamma', 0.5)
        )
        self.current_epoch = 0
        self.lr_history = []
        self.train_losses = []
        self.val_accuracies = []
        self.quantum_usage = []
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.quantum_strength = 1.0

    def _should_use_quantum(self, epoch: int, val_acc: float) -> bool:
        if self.quantum_strategy == 'adaptive':
            if epoch < 2:
                return True
            elif val_acc > 70:
                return np.random.random() < 0.2
            elif val_acc > 60:
                return np.random.random() < 0.5
            else:
                return np.random.random() < 0.8
        return True

    def _get_quantum_strength(self, epoch: int, val_acc: float) -> float:
        if self.quantum_strategy == 'adaptive':
            if epoch < 2:
                return 1.0
            elif val_acc > 70:
                return 0.3
            elif val_acc > 60:
                return 0.6
            else:
                return 0.9
        else:
            return 1.0

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor, val_acc: float = 0.0) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        use_quantum = self._should_use_quantum(self.current_epoch, val_acc)
        quantum_strength = self._get_quantum_strength(self.current_epoch, val_acc)
        if use_quantum:
            self.model.quantum_backward(loss)
            if quantum_strength < 1.0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and "fc2" in name:
                            classical_grad = param.grad.clone()
                            self.model.quantum_backward(loss)
                            quantum_grad = param.grad.clone()
                            param.grad = quantum_strength * quantum_grad + (1 - quantum_strength) * classical_grad
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.quantum_usage.append(1.0 if use_quantum else 0.0)
        return loss.item()

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return val_loss / len(val_loader), 100. * correct / total

    def step_scheduler(self, val_loss: Optional[float] = None):
        if self.scheduler is not None:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)

    def get_current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
        
    def get_training_stats(self) -> dict:
        return {
            'train_losses': self.train_losses.copy(),
            'val_accuracies': self.val_accuracies.copy(),
            'learning_rates': self.lr_history.copy(),
            'quantum_usage': self.quantum_usage.copy(),
            'current_epoch': self.current_epoch,
            'best_val_acc': self.best_val_acc
        }

def train_hybrid(num_epochs=30, batch_size=50):
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
    model = QuantumEnhancedResNet20(num_classes=10, quantum_optimizer_depth=3).to(device)
    trainer = AdaptiveQuantumTrainer(
        model=model,
        device=device,
        initial_lr=0.001,
        scheduler_params={'step_size': 5, 'gamma': 0.8},
        quantum_strategy='adaptive'
    )
    print(f"Initial Learning Rate: {trainer.get_current_lr():.6f}")
    print(f"Quantum Strategy: adaptive")
    print(f"Total Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print("\n" + "="*70)
    print("TRAINING PROGRESS")
    print("="*70)
    # Create results directories if they don't exist
    os.makedirs('results/npy', exist_ok=True)
    os.makedirs('results/images', exist_ok=True)
    for epoch in range(num_epochs):
        trainer.current_epoch = epoch
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            current_val_acc = trainer.val_accuracies[-1] if trainer.val_accuracies else 0.0
            loss = trainer.train_step(inputs, targets, current_val_acc)
            epoch_loss += loss
            batch_count += 1
            if batch_idx % 100 == 0:
                current_lr = trainer.get_current_lr()
                quantum_used = trainer.quantum_usage[-1] if trainer.quantum_usage else 0.0
                print(f'Epoch: {epoch+1:2d}/{num_epochs} | Batch: {batch_idx:4d}/{len(trainloader):4d} | '
                      f'Loss: {loss:.4f} | LR: {current_lr:.6f} | Quantum: {quantum_used:.1f}')
        val_loss, val_acc = trainer.validate(testloader)
        trainer.step_scheduler(val_loss)
        avg_train_loss = epoch_loss / batch_count
        current_lr = trainer.get_current_lr()
        trainer.train_losses.append(avg_train_loss)
        trainer.val_accuracies.append(val_acc)
        if val_acc > trainer.best_val_acc:
            trainer.best_val_acc = val_acc
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
        quantum_usage_epoch = np.mean(trainer.quantum_usage[-len(trainloader):]) if trainer.quantum_usage else 0.0
        print(f'Epoch: {epoch+1:2d}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f} | '
              f'Quantum Usage: {quantum_usage_epoch:.1%}')
    # Final metrics
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
    f1 = f1_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    print(f'Final F1 Score: {f1:.4f}')
    print(f'Final Recall Score: {recall:.4f}')
    print(f'Final Confusion Matrix:\n{cm}')
    np.save('results/npy/hybrid_f1_scores.npy', np.array([f1]))
    np.save('results/npy/hybrid_recall_scores.npy', np.array([recall]))
    np.save('results/npy/hybrid_confusion_matrices.npy', np.array([cm]))
    np.save('results/npy/hybrid_train_losses.npy', np.array(trainer.train_losses))
    np.save('results/npy/hybrid_val_accuracies.npy', np.array(trainer.val_accuracies))
    np.save('results/npy/hybrid_lr_history.npy', np.array(trainer.lr_history))
    # Plot results
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, trainer.train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs, trainer.val_accuracies, 'r-s', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Hybrid Model Training (Adaptive Quantum)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(epochs, trainer.lr_history, 'g-^', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Step LR Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/images/hybrid_train_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved to 'results/images/hybrid_train_results.png'")

if __name__ == "__main__":
    train_hybrid() 