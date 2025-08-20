import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import DiffusionUNet
from diffusion import DiffusionModel
from Loss import CombinedLoss
from dataset import MedicalImageDataset, create_dataloaders
from Utils import count_parameters
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import train_step, visualize_results, process_multiple_images


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionUNet().to(device)
    diffusion = DiffusionModel(noise_steps=10000)
    loss_fn = CombinedLoss(
        l1_weight=0.3,
        perceptual_weight=0.7
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Create train and validation dataloaders
    data_dir = "Task1"
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        train_split=0.8,
        num_workers=4
    )
    
    # Print model statistics
    param_info = count_parameters(model)
    print("\nModel Statistics:")
    for key, value in param_info.items():
        print(f"{key}: {value}")

    # Training loop
    epochs = 2
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        
        for batch in progress_bar:
            loss = train_step(model, diffusion, batch, optimizer, loss_fn, device, scaler)
            epoch_loss += loss
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_batch_loss = train_step(model, diffusion, batch, optimizer, loss_fn, device, scaler, training=False)
                val_loss += val_batch_loss
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f'checkpoint_epoch_{epoch + 1}.pt')
    
    # Save the final model
    save_path = "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'diffusion_noise_steps': diffusion.noise_steps,
        'epoch': epochs,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss
    }, save_path)
    print(f"Model saved to {save_path}")

    # Evaluation phase
    print("\nStarting evaluation phase...")
    model.eval()
    
    # Use validation dataset for final evaluation
    results, metrics = process_multiple_images(model, diffusion, val_dataset, num_images=6, device=device)    
    visualize_results(results, metrics)

if __name__ == "__main__":
    main()