"""Train binary forest segmentation U-Net and export loss/validation curves."""
from __future__ import annotations
import argparse, csv, json, random
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
        x=np.nan_to_num(np.clip(x,0.,1.),nan=0.,posinf=1.,neginf=0.).astype(np.float32)
        x=np.moveaxis(x,0,-1)
        if self.augment:
            z=self.tf(image=x,mask=y); x,y=z['image'],z['mask']
        x=torch.from_numpy(np.ascontiguousarray(np.moveaxis(x,-1,0))).float()
        y=torch.from_numpy(np.ascontiguousarray(y[None])).float()
        return x,y

class BCEDice(nn.Module):
    def __init__(self): super().__init__(); self.bce=nn.BCEWithLogitsLoss()
    def forward(self,logits,target):
        bce=self.bce(logits,target); prob=torch.sigmoid(logits); dims=(1,2,3); smooth=1.
        dice=1-((2*(prob*target).sum(dims)+smooth)/((prob+target).sum(dims)+smooth)).mean()
        return .5*bce+.5*dice

def main():
    p=argparse.ArgumentParser(); p.add_argument('--patch-dir',default='data/processed/patches'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--in-channels',type=int,required=True); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=17); p.add_argument('--out',default='outputs/best_model.pt'); p.add_argument('--threshold',type=float,default=.5); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=Path(a.patch_dir); manifest=json.loads((root/'splits.json').read_text())
    ds={s:PatchDataset(root,manifest['splits'][s],s=='train') for s in ('train','val')}
    if not len(ds['train']) or not len(ds['val']): raise ValueError('Train and validation splits must both contain patches.')
    loaders={s:DataLoader(ds[s],batch_size=a.batch_size,shuffle=s=='train',num_workers=a.workers,pin_memory=device.type=='cuda') for s in ds}
    model=smp.Unet(encoder_name='resnet34',encoder_weights='imagenet',in_channels=a.in_channels,classes=1,activation=None).to(device)
    criterion=BCEDice(); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); best=float('inf'); history=[]
    for epoch in range(1,a.epochs+1):
        row={'epoch':epoch}
        for phase in ('train','val'):
            model.train(phase=='train'); total=tp=fp=fn=0
            for x,y in loaders[phase]:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase=='train'):
                    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                        logits=model(x); loss=criterion(logits,y)
                    if phase=='train': scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                total+=loss.item()*x.size(0)
                pred=torch.sigmoid(logits.detach())>=a.threshold; truth=y>=.5
                tp+=int((pred&truth).sum().item()); fp+=int((pred&~truth).sum().item()); fn+=int((~pred&truth).sum().item())
            row[f'{phase}_loss']=total/len(ds[phase])
            precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
            row[f'{phase}_precision']=precision; row[f'{phase}_recall']=recall
            row[f'{phase}_f1']=2*precision*recall/max(1e-12,precision+recall)
            row[f'{phase}_iou']=tp/max(1,tp+fp+fn)
        history.append(row)
        print(f"epoch {epoch:03d}/{a.epochs} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} val_iou={row['val_iou']:.4f} val_f1={row['val_f1']:.4f}",flush=True)
        with (out.parent/'training_history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        if row['val_loss']<best:
            best=row['val_loss']; torch.save({'model':model.state_dict(),'in_channels':a.in_channels,'encoder':'resnet34','epoch':epoch,'val_loss':best,'threshold':a.threshold},out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep=[r['epoch'] for r in history]
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    ax[0].plot(ep,[r['train_loss'] for r in history],label='Train'); ax[0].plot(ep,[r['val_loss'] for r in history],label='Validation'); ax[0].set(title='Loss by epoch',xlabel='Epoch',ylabel='BCE + Dice loss'); ax[0].grid(alpha=.25); ax[0].legend()
    ax[1].plot(ep,[r['val_iou'] for r in history],label='Validation IoU'); ax[1].plot(ep,[r['val_f1'] for r in history],label='Validation F1'); ax[1].plot(ep,[r['val_precision'] for r in history],label='Validation precision',alpha=.75); ax[1].plot(ep,[r['val_recall'] for r in history],label='Validation recall',alpha=.75); ax[1].set(title=f'Validation metrics (threshold={a.threshold:g})',xlabel='Epoch',ylabel='Score'); ax[1].set_ylim(0,1); ax[1].grid(alpha=.25); ax[1].legend()
    fig.tight_layout(); fig.savefig(out.parent/'training_curves.png',dpi=160); plt.close(fig)
    print('best checkpoint:',out); print('history:',out.parent/'training_history.csv'); print('curves:',out.parent/'training_curves.png')
if __name__=='__main__': main()
