# PyTorch AMP Deprecation Warning Fix

## Issue
When running the training notebook, you may see these warnings:
```
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
```

## Explanation
These are deprecation warnings, not errors. The code will still run correctly, but PyTorch has updated the API for Automatic Mixed Precision (AMP) training.

### Old API (deprecated)
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
with autocast():
    # training code
```

### New API (recommended)
```python
from torch.amp import GradScaler, autocast

scaler = GradScaler('cuda')
with autocast('cuda'):
    # training code
```

## Fix for the Training Cell

Replace the import line and update the usage:

```python
import time
from torch.amp import GradScaler, autocast  # Updated import
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# ... (T4 optimizations code remains the same) ...

def train_model(model_name, model, train_loader, config):
    """T4-optimized training function"""
    print(f"\n🚀 Training {model_name}...")
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    # Mixed precision setup
    scaler = GradScaler('cuda') if config['mixed_precision'] else None  # Updated
    
    # Training history
    history = {'train_loss': []}
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)
            
            # Mixed precision training
            if config['mixed_precision']:
                with autocast('cuda'):  # Updated
                    reconstructed = model(images)
                    loss = criterion(reconstructed, images)
                
                # Backward pass with scaler
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # ... (rest of the training code remains the same) ...
```

## Key Changes

1. **Import statement**: 
   - Old: `from torch.cuda.amp import GradScaler, autocast`
   - New: `from torch.amp import GradScaler, autocast`

2. **GradScaler initialization**:
   - Old: `GradScaler()`
   - New: `GradScaler('cuda')`

3. **Autocast context**:
   - Old: `with autocast():`
   - New: `with autocast('cuda'):`

## Note
This is just a warning and won't affect the training process. The old API still works in current PyTorch versions but may be removed in future releases. It's recommended to update to the new API for future compatibility.