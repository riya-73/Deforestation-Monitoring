"""Evaluate a saved model and save a side-by-side prediction overlay."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

def main():
 p=argparse.ArgumentParser(); p.add_argument('--patch-dir',default='data/processed/patches'); p.add_argument('--checkpoint',default='outputs/best_model.pt'); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--out',default='outputs/evaluation'); a=p.parse_args()
 root=Path(a.patch_dir); split=json.loads((root/'splits.json').read_text())['splits']['test']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ck=torch.load(a.checkpoint,map_location=dev,weights_only=False); model=smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=ck['in_channels'],classes=1,activation=None).to(dev); model.load_state_dict(ck['model']); model.eval()
 tp=fp=fn=tn=0; example=None
 with torch.no_grad():
  for name in split:
   with np.load(root/name) as z: x=z['image'].copy(); y=z['mask'].copy()
   x=np.nan_to_num(np.clip(x,0,1),nan=0,posinf=1,neginf=0).astype(np.float32)
   prob=torch.sigmoid(model(torch.from_numpy(x[None]).to(dev))).cpu().numpy()[0,0]; pred=prob>=a.threshold; truth=y==1
   tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum()); fn+=int((~pred&truth).sum()); tn+=int((~pred&~truth).sum())
   if example is None: example=(x,y,pred)
 precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); f1=2*precision*recall/max(1e-12,precision+recall); iou=tp/max(1,tp+fp+fn)
 metrics={'threshold':a.threshold,'precision':precision,'recall':recall,'f1':f1,'iou':iou,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'test_patches':len(split)}
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
 if example:
  x,y,pred=example
  # First three bands displayed as RGB only if channels are known to be B/G/R.
  rgb=np.moveaxis(x[:3],0,-1); lo,hi=np.percentile(rgb,[2,98]); rgb=np.clip((rgb-lo)/(hi-lo+1e-6),0,1)
  fig,axs=plt.subplots(1,3,figsize=(12,4)); axs[0].imshow(rgb); axs[0].set_title('Input (first 3 bands)'); axs[1].imshow(y,cmap='gray',vmin=0,vmax=1); axs[1].set_title('Ground truth'); axs[2].imshow(pred,cmap='gray',vmin=0,vmax=1); axs[2].set_title('Prediction')
  for ax in axs: ax.axis('off')
  fig.tight_layout(); fig.savefig(out/'test_overlay.png',dpi=160); plt.close(fig)
if __name__=='__main__': main()
