import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from quantum_optimizer import QuantumErrorOptimizer
from typing import Tuple, Optional, Literal

class QuantumEnhancedCNN(nn.Module):
    def __init__(self, num_classes: int = 10, quantum_optimizer_depth: int = 3):
        """
        Initialize the Quantum-Enhanced CNN.
        
        Args:
            num_classes (int): Number of output classes
            quantum_optimizer_depth (int): Depth of the quantum circuit for optimization
        """
        super(QuantumEnhancedCNN, self).__init__()
        
        # Classical CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

        # Quantum optimizer for error optimization
        self.quantum_optimizer = QuantumErrorOptimizer(num_qubits=4, depth=quantum_optimizer_depth)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def quantum_backward(self, loss: torch.Tensor):
        """
        Custom backward pass using quantum-enhanced error optimization.
        Only applies quantum optimization to the final layer (fc2) to preserve
        gradient information in the backbone layers.
        
        Args:
            loss (torch.Tensor): The computed loss
        """
        # Compute classical gradients
        loss.backward(retain_graph=True)
        
        # Optimize gradients using quantum circuit only for the final layer
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if "fc2" in name:  # Only apply to final layer
                        # Convert gradient to numpy for quantum processing
                        grad_np = param.grad.cpu().numpy().flatten()
                        
                        # Apply quantum optimization
                        optimized_grad = self.quantum_optimizer.optimize_error(grad_np)
                        
                        # Convert back to torch tensor with correct dtype and device
                        optimized_grad = torch.from_numpy(
                            optimized_grad.reshape(param.grad.shape)
                        ).to(dtype=param.grad.dtype, device=param.grad.device)
                        
                        # Update gradients with quantum-optimized version
                        param.grad = optimized_grad

class QuantumEnhancedTrainer:
    def __init__(self, 
                 model: QuantumEnhancedCNN, 
                 device: torch.device, 
                 quantum_epochs: int = 2,
                 initial_lr: float = 0.001,
                 scheduler_type: Literal['step', 'exponential', 'cosine', 'plateau', 'none'] = 'step',
                 scheduler_params: Optional[dict] = None):
        """
        Initialize the trainer for the Quantum-Enhanced CNN with dynamic learning rate.
        
        Args:
            model (QuantumEnhancedCNN): The quantum-enhanced CNN model
            device (torch.device): The device to train on
            quantum_epochs (int): Number of epochs to use quantum optimization
            initial_lr (float): Initial learning rate
            scheduler_type (str): Type of learning rate scheduler
                - 'step': Step decay scheduler
                - 'exponential': Exponential decay scheduler
                - 'cosine': Cosine annealing scheduler
                - 'plateau': Reduce on plateau scheduler
                - 'none': No scheduler (constant learning rate)
            scheduler_params (dict): Parameters for the scheduler
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
        self.quantum_epochs = quantum_epochs
        self.current_epoch = 0
        self.scheduler_type = scheduler_type
        
        # Set default scheduler parameters if not provided
        if scheduler_params is None:
            scheduler_params = self._get_default_scheduler_params(scheduler_type)
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(scheduler_type, scheduler_params)
        
        # Track learning rate history
        self.lr_history = []
        
    def _get_default_scheduler_params(self, scheduler_type: str) -> dict:
        """Get default parameters for different scheduler types."""
        defaults = {
            'step': {'step_size': 2, 'gamma': 0.5},  # Optimized for better performance
            'exponential': {'gamma': 0.95},
            'cosine': {'T_max': 100, 'eta_min': 1e-6},
            'plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10, 'verbose': True},
            'none': {}
        }
        return defaults.get(scheduler_type, {})
    
    def _create_scheduler(self, scheduler_type: str, params: dict):
        """Create the learning rate scheduler based on type."""
        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=params.get('step_size', 30), 
                gamma=params.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, 
                gamma=params.get('gamma', 0.95)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=params.get('T_max', 100), 
                eta_min=params.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 10),
                verbose=params.get('verbose', True)
            )
        else:  # 'none'
            return None
    
    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def step_scheduler(self, val_loss: Optional[float] = None):
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            if self.scheduler_type == 'plateau' and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler_type != 'plateau':
                self.scheduler.step()
            
            # Record current learning rate
            current_lr = self.get_current_lr()
            self.lr_history.append(current_lr)
    
    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        
        # Use quantum optimization only for the first N epochs
        if self.current_epoch < self.quantum_epochs:
            self.model.quantum_backward(loss)
        else:
            loss.backward()
            
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
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
    
    def get_lr_history(self) -> list:
        """Get the learning rate history."""
        return self.lr_history.copy()

# --- ResNet-20 for CIFAR-10 ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class QuantumEnhancedResNet20(ResNet20):
    def __init__(self, num_classes=10, quantum_optimizer_depth=3):
        super().__init__(num_classes=num_classes)
        self.quantum_optimizer = QuantumErrorOptimizer(num_qubits=4, depth=quantum_optimizer_depth)
    def quantum_backward(self, loss: torch.Tensor):
        loss.backward(retain_graph=True)
        with torch.no_grad():
            for name, param in self.named_parameters(): 
                if param.grad is not None:
                    if "fc" in name:  # Only apply to final layer
                        grad_np = param.grad.cpu().numpy().flatten()
                        optimized_grad = self.quantum_optimizer.optimize_error(grad_np)
                        optimized_grad = torch.from_numpy(
                            optimized_grad.reshape(param.grad.shape)
                        ).to(dtype=param.grad.dtype, device=param.grad.device)
                        param.grad = optimized_grad 