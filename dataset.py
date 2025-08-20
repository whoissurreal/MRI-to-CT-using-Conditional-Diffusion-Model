import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import nibabel as nib



class MedicalImageDataset(Dataset):
    """
    Looks for e.g.:
      /data/brain/<sample>/{mr.nii.gz, ct.nii.gz}
      /data/pelvis/<sample>/{mr.nii.gz, ct.nii.gz}
    For each sample, we can read slices from the volume (slices_per_volume=150).
    """
    def __init__(self, root_dir, target_size=(224,224), slices_per_volume=150):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.slices_per_volume = slices_per_volume
        self.pairs = []

        for region in ['brain','pelvis']:
            region_path = self.root_dir/region
            if not region_path.is_dir():
                continue
            for folder in region_path.iterdir():
                if not folder.is_dir():
                    continue
                ct_file = folder/'ct.nii.gz'
                mr_file = folder/'mr.nii.gz'
                if ct_file.exists() and mr_file.exists():
                    self.pairs.append((mr_file, ct_file))

    def __len__(self):
        return len(self.pairs)*self.slices_per_volume

    def __getitem__(self,idx):
        pair_idx = idx//self.slices_per_volume
        slice_idx= idx%self.slices_per_volume
        mr_path, ct_path = self.pairs[pair_idx]

        mr_data = self._load_and_process(mr_path, slice_idx)
        ct_data = self._load_and_process(ct_path, slice_idx)
        return {'mri': mr_data, 'ct': ct_data}

    def _load_and_process(self, path, slice_idx):
        img = nib.load(path)
        data= img.get_fdata()  # (H,W,Slices) or something
        total_slices = data.shape[-1]

        if total_slices>=self.slices_per_volume:
            # pick evenly spaced
            indices = np.linspace(0, total_slices-1, self.slices_per_volume,dtype=int)
            sidx = indices[slice_idx]
        else:
            sidx = slice_idx%total_slices

        slice_data = data[..., int(sidx)]
        slice_data = (slice_data - slice_data.min())/(slice_data.max()-slice_data.min()+1e-8)
        # shape => (1,H,W)
        t = torch.FloatTensor(slice_data).unsqueeze(0)
        # resize
        t = F.interpolate(t.unsqueeze(0), size=self.target_size,
                          mode='bilinear', align_corners=False).squeeze(0)
        return t

def create_dataloaders(data_dir,
                       batch_size=4,
                       train_split=0.8,
                       num_workers=2):
    dataset = MedicalImageDataset(data_dir, target_size=(224,224), slices_per_volume=150)
    train_size = int(train_split*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size,val_size])

    train_loader=DataLoader(train_dataset,batch_size=batch_size,
                            shuffle=True,num_workers=num_workers,pin_memory=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,
                          shuffle=False,num_workers=num_workers,pin_memory=True)
    return train_loader,val_loader,train_dataset,val_dataset




