import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt

def train_step(model, diffusion, batch, optimizer, loss_fn, device, scaler=None):
    """Enhanced training step with gradient clipping and logging"""
    optimizer.zero_grad()
    
    mri = batch['mri'].to(device)
    ct = batch['ct'].to(device)
    
    # Mix-up augmentation
    if np.random.random() < 0.3:
        lam = np.random.beta(0.2, 0.2)
        rand_index = torch.randperm(mri.size(0))
        mri = lam * mri + (1 - lam) * mri[rand_index]
        ct = lam * ct + (1 - lam) * ct[rand_index]
    
    t = diffusion.sample_timesteps(mri.shape[0]).to(device)
    noisy_mri, noise = diffusion.noise_images(mri, t)
    
    if scaler is not None:
        with torch.cuda.amp.autocast():
            pred_noise = model(noisy_mri, t, ct)
            loss = loss_fn(pred_noise, noise)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
    else:
        pred_noise = model(noisy_mri, t, ct)
        loss = loss_fn(pred_noise, noise)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Update EMA if enabled
    if hasattr(model, 'use_ema') and model.use_ema:
        model.ema.update(model)
    
    return loss.item()

def process_multiple_images(model, diffusion, dataset, num_images=6, device='cuda'):
    """
    Process multiple images through the diffusion model
    """
    model.eval()
    results = []
    metrics = []
    
    # Create a dataloader with batch_size=1 to process images one by one
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_images:
                break
                
            mri = batch['mri'].to(device)
            ct = batch['ct'].to(device)
            
            # Generate CT from MRI
            t = torch.tensor([0], device=device)
            noisy_mri, _ = diffusion.noise_images(mri, t)
            generated_ct = model(noisy_mri, t, ct)
            
            # Store results
            results.append({
                'mri': mri.squeeze().cpu().numpy(),
                'ct': ct.squeeze().cpu().numpy(),
                'generated_ct': generated_ct.squeeze().cpu().numpy()
            })
            
            # Calculate metrics
            ct_np = ct.squeeze().cpu().numpy()
            gen_ct_np = generated_ct.squeeze().cpu().numpy()
            
            # Calculate PSNR
            psnr = calculate_psnr(ct_np, gen_ct_np, data_range=gen_ct_np.max() - gen_ct_np.min())
            
            # Calculate SSIM
            ct_norm = (ct_np - ct_np.min()) / (ct_np.max() - ct_np.min() + 1e-8)
            gen_ct_norm = (gen_ct_np - gen_ct_np.min()) / (gen_ct_np.max() - gen_ct_np.min() + 1e-8)
            
            ct_tensor = torch.tensor(ct_norm).unsqueeze(0).unsqueeze(0)
            gen_ct_tensor = torch.tensor(gen_ct_norm).unsqueeze(0).unsqueeze(0)
            
            ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
            ssim = ssim_module(gen_ct_tensor, ct_tensor).item()
            
            metrics.append({
                'psnr': psnr,
                'ssim': ssim
            })
    
    return results, metrics

def visualize_results(results, metrics):
    """
    Create a grid visualization of the results with metrics
    """
    num_images = len(results)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    for i in range(num_images):
        # Get current row of results
        mri = results[i]['mri']
        ct = results[i]['ct']
        generated_ct = results[i]['generated_ct']
        
        # Plot MRI
        axes[i, 0].imshow(mri, cmap='gray')
        axes[i, 0].set_title(f'MRI {i+1}')
        axes[i, 0].axis('off')
        
        # Plot original CT
        axes[i, 1].imshow(ct, cmap='gray')
        axes[i, 1].set_title(f'Original CT {i+1}')
        axes[i, 1].axis('off')
        
        # Plot generated CT with metrics
        axes[i, 2].imshow(generated_ct, cmap='gray')
        axes[i, 2].set_title(f'Generated CT {i+1}\nPSNR: {metrics[i]["psnr"]:.2f}dB\nSSIM: {metrics[i]["ssim"]:.4f}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print average metrics
    avg_psnr = np.mean([m['psnr'] for m in metrics])
    avg_ssim = np.mean([m['ssim'] for m in metrics])
    print(f"\nAverage Metrics across {num_images} images:")
    print(f"Average PSNR: {avg_psnr:.2f}dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
