"""Train binary forest segmentation U-Net with BCE + soft Dice loss."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

class PatchDataset(Dataset):
    def __init__(self, root, files, augment=False):
        self.root=Path(root); self.files=files; self.augment=augment
        if augment:
            import albumentations as A
            self.tf=A.Compose([A.HorizontalFlip(p=.5),A.VerticalFlip(p=.5),A.RandomRotate90(p=.5)])
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        with np.load(self.root/self.files[i]) as z: x=z['image'].copy(); y=z['mask'].copy()
        # Expected surface reflectance [0,1]. Clipping limits rare outliers.
        # If source uses scaled integers or another normalization, correct upstream.
        x=np.clip(x,0.,1.).astype(np.float32)
        x=np.nan_to_num(x,nan=0.,posinf=1.,neginf=0.)
        x=np.moveaxis(x,0,-1)
        if self.augment:
            out=self.tf(image=x,mask=y); x,y=out['image'],out['mask']
        x=torch.from_numpy(np.ascontiguousarray(np.moveaxis(x,-1,0))).float()
        y=torch.from_numpy(np.ascontiguousarray(y[None])).float()
        return x,y

class BCEDice(nn.Module):
    def __init__(self): super().__init__(); self.bce=nn.BCEWithLogitsLoss()
    def forward(self,logits,target):
        bce=self.bce(logits,target); probs=torch.sigmoid(logits)
        dims=(1,2,3); smooth=1.
        dice=1-((2*(probs*target).sum(dims)+smooth)/((probs+target).sum(dims)+smooth)).mean()
        return .5*bce+.5*dice

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patch-dir',default='data/processed/patches'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--in-channels',type=int,required=True); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=17); p.add_argument('--out',default='outputs/best_model.pt'); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path(a.patch_dir); manifest=json.loads((root/'splits.json').read_text())
    ds={s:PatchDataset(root,manifest['splits'][s],s=='train') for s in ('train','val')}
    loaders={s:DataLoader(ds[s],batch_size=a.batch_size,shuffle=s=='train',num_workers=a.workers,pin_memory=device.type=='cuda') for s in ds}
    model=smp.Unet(encoder_name='resnet34',encoder_weights='imagenet',in_channels=a.in_channels,classes=1,activation=None).to(device)
    criterion=BCEDice(); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); best=float('inf')
    for epoch in range(1,a.epochs+1):
        row={}
        for phase in ('train','val'):
            model.train(phase=='train'); total=0.
            for x,y in loaders[phase]:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'):
                    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                        logits=model(x); loss=criterion(logits,y)
                    if phase=='train':
                        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                total+=loss.item()*x.size(0)
            row[phase]=total/max(1,len(ds[phase]))
        print(f"epoch {epoch:03d}/{a.epochs} train={row['train']:.4f} val={row['val']:.4f}")
        if row['val']<best:
            best=row['val']; torch.save({'model':model.state_dict(),'in_channels':a.in_channels,'encoder':'resnet34','epoch':epoch,'val_loss':best},out)
    print('best checkpoint:',out)
if __name__=='__main__': main()
