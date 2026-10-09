"""Create labeled patches and geographically isolated train/val/test splits."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


def prepare(image_path, label_path, out_dir, patch_size=256, seed=17, fractions=(.70,.15,.15), min_valid=0.95):
    out=Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng=random.Random(seed)
    with rasterio.open(image_path) as im, rasterio.open(label_path) as lab:
        if im.crs is None or lab.crs is None:
            raise ValueError('Both inputs need a CRS. A .npy file has no CRS/transform; georeference it first using trusted source metadata.')
        if (im.crs != lab.crs or im.transform != lab.transform or im.width != lab.width or im.height != lab.height):
            raise ValueError('Image and label must already share the exact grid. Warp the categorical label to image grid with nearest-neighbor first.')
        records=[]; H,W=im.height,im.width
        # Patch grid is also the split unit; assigned blocks never cross splits.
        for r in range(0,H-patch_size+1,patch_size):
            for c in range(0,W-patch_size+1,patch_size):
                win=Window(c,r,patch_size,patch_size)
                x=im.read(window=win).astype(np.float32)
                y=lab.read(1,window=win)
                valid=np.isin(y,[0,1])
                if valid.mean()<min_valid or not np.isfinite(x).all(): continue
                # Store unnormalized reflectance; training scales/clips consistently.
                group=f'block_{r//(patch_size*4):04d}_{c//(patch_size*4):04d}'
                name=f'p_r{r:05d}_c{c:05d}.npz'
                np.savez_compressed(out/name, image=x, mask=y.astype(np.uint8), row=r, col=c, group=group)
                records.append({'file':name,'group':group,'row':r,'col':c})
    if len(records)<3: raise ValueError(f'Only {len(records)} valid patches; check label nodata, grid, and AOI size.')
    groups=sorted({r['group'] for r in records}); rng.shuffle(groups)
    # Randomize whole geographic blocks, never individual adjacent patches.
    n=len(groups); ntrain=max(1,int(n*fractions[0])); nval=max(1,int(n*fractions[1]))
    if ntrain+nval>=n: ntrain=max(1,n-2); nval=1
    split_groups={'train':set(groups[:ntrain]),'val':set(groups[ntrain:ntrain+nval]),'test':set(groups[ntrain+nval:])}
    manifest={'patch_size':patch_size,'seed':seed,'splits':{},'source_image':str(image_path),'source_label':str(label_path)}
    for s,gs in split_groups.items():
        manifest['splits'][s]=[r['file'] for r in records if r['group'] in gs]
    (out/'splits.json').write_text(json.dumps(manifest,indent=2))
    print('patches:',len(records),'geographic blocks:',n,'split counts:',{k:len(v) for k,v in manifest['splits'].items()})
    print('split manifest:',out/'splits.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--image',required=True); p.add_argument('--label',required=True); p.add_argument('--out',default='data/processed/patches'); p.add_argument('--patch-size',type=int,default=256); p.add_argument('--seed',type=int,default=17); p.add_argument('--min-valid',type=float,default=.95)
    a=p.parse_args(); prepare(a.image,a.label,a.out,a.patch_size,a.seed,min_valid=a.min_valid)
