# ═════════════════════════════════════════════════════════════════════
#  4D‑STEM unsupervised denoiser  – 10‑cycle refinement (Sadri et al.)
#  Works on free‑tier Colab GPU • 2025‑07‑26
# ═════════════════════════════════════════════════════════════════════

# ---------- Drive (holds .npy files) ----------
import torch as th
device = th.device('cuda:1')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # just in case
import torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Available GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}") 

# ═════════════════════════════════════════════════════════════════════
#  SrTiO3 4D‑STEM  • 10‑cycle self‑supervised denoiser  (Sadri et al.)
#  Colab / T4‑friendly  – 2025‑07‑26
# ═════════════════════════════════════════════════════════════════════

import torch, os, pathlib, numpy as np, matplotlib.pyplot as plt, gc
torch.set_float32_matmul_precision("medium")



# -------- EDIT YOUR OWN PATHS ---------------------------------------
LOW_PATH  = '/home/dasun/initial/my_project/Denoise/Data/03_denoising_SrTiO3_High_mag_Low_dose.npy'
HIGH_PATH = '/home/dasun/initial/my_project/Denoise/Data/03_denoising_SrTiO3_High_mag_High_dose.npy'
OUT_DIR   = '/home/dasun/initial/my_project/Denoise/Data'
# --------------------------------------------------------------------

device = torch.device('cuda')
print('GPU:', torch.cuda.get_device_name(0))
# ═════════════ raw data ═════════════════════════════════════════════
raw4d = np.load(LOW_PATH ).astype(np.float32)            # untouched copy
hi4d  = np.load(HIGH_PATH).astype(np.float32)            # for comparison
scan_H, scan_W, P, _ = raw4d.shape
NB_OFF = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

# ═════════════ dataset helper ═══════════════════════════════════════
from torch.utils.data import Dataset, DataLoader
class FourDSTEM(Dataset):
    def __init__(self, cube):
        self.A = cube.reshape(-1, P, P)
        self.scan = scan_H
        self.inner= scan_H - 2
    def __len__(self): return self.inner**2
    def __getitem__(self, idx):
        i = idx // self.inner + 1
        j = idx %  self.inner + 1
        neigh = [self.A[(i+di)*self.scan + (j+dj)] for di,dj in NB_OFF]
        centre=  self.A[i*self.scan + j]
        return (torch.from_numpy(np.stack(neigh)).float(),
                torch.from_numpy(centre).float().unsqueeze(0))
def make_loader(cube):
    return DataLoader(FourDSTEM(cube), batch_size=32,
                      shuffle=True, num_workers=0, pin_memory=True)

# ═════════════ model & lightning ════════════════════════════════════
import torch.nn as nn, torch.nn.functional as F, pytorch_lightning as pl
def blk(ci,co):
    return nn.Sequential(nn.Conv2d(ci,co,3,1,1), nn.ReLU(),
                         nn.Conv2d(co,co,3,1,1), nn.ReLU())
class UNet(nn.Module):
    def __init__(self, in_ch=8, base=24):
        super().__init__()
        self.e1=blk(in_ch,base)
        self.e2=blk(base,base*2);  self.p2=nn.MaxPool2d(2)
        self.e3=blk(base*2,base*4);self.p3=nn.MaxPool2d(2)
        self.e4=blk(base*4,base*8);self.p4=nn.MaxPool2d(2)
        self.bot=blk(base*8,base*16)
        self.u3=nn.ConvTranspose2d(base*16,base*8,2,2); self.d3=blk(base*16,base*8)
        self.u2=nn.ConvTranspose2d(base*8,base*4,2,2);  self.d2=blk(base*8,base*4)
        self.u1=nn.ConvTranspose2d(base*4,base*2,2,2);  self.d1=blk(base*4,base*2)
        self.u0=nn.ConvTranspose2d(base*2,base,2,2);    self.d0=blk(base*2,base)
        self.out=nn.Conv2d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p2(e1)); e3=self.e3(self.p3(e2))
        e4=self.e4(self.p4(e3)); b=self.bot(self.p4(e4))
        d3=self.d3(torch.cat([self.u3(b),e4],1))
        d2=self.d2(torch.cat([self.u2(d3),e3],1))
        d1=self.d1(torch.cat([self.u1(d2),e2],1))
        d0=self.d0(torch.cat([self.u0(d1),e1],1))
        return torch.square(self.out(d0))            # ≥0

class LitDenoiser(pl.LightningModule):
    def __init__(self, warm=4, lr=1e-3):
        super().__init__(); self.net=UNet(); self.warm=warm; self.lr=lr
        self.pnll=nn.PoissonNLLLoss(log_input=False, reduction='mean')
    def forward(self,x): return self.net(x)
    def training_step(self,batch,_):
        nb,tgt=batch; pred=self(nb)
        if self.current_epoch<self.warm:
            z=lambda t:(t-t.mean())/(t.std()+1e-6)
            loss=F.mse_loss(z(pred),z(tgt))
        else:
            loss=self.pnll(torch.clamp(pred,1e-8),tgt)
            PAC_pred = pred.mean((2,3)); PAC_tgt = tgt.mean((2,3))
            BF_pred  = pred.sum((2,3));  BF_tgt  = tgt.sum((2,3))
            loss += 0.02*F.mse_loss(PAC_pred,PAC_tgt)
            loss += 0.01*F.mse_loss(BF_pred ,BF_tgt )
        self.log('loss',loss,prog_bar=True); return loss
    def configure_optimizers(self):
        opt=torch.optim.Adam(self.parameters(),lr=self.lr)
        return {"optimizer": opt}

# ═════════════ iterative refinement (10 cycles) ═════════════════════
den4d = raw4d.copy()                                   # seed
model = None                                           # placeholder

for c in range(1, 11):
    β = c / 10.0
    print(f'\n── Cycle {c}/10  (β={β:.1f}) ──')

    # 1) blended input for this cycle
    mix4d = (1-β)*raw4d + β*den4d

    # 2) (re)build LightningModule and copy previous weights
    new_model = LitDenoiser().to(device)
    if model is not None:
        new_model.net.load_state_dict(model.net.state_dict())
        del model; torch.cuda.empty_cache()
    model = new_model

    # 3) train 15 epochs (4 MSE + 11 Poisson)
    trainer = pl.Trainer(max_epochs=15, accelerator='gpu', devices=1,
                         precision=32, enable_progress_bar=True)
    trainer.fit(model, make_loader(mix4d))

    # 4) full‑cube inference
    model.eval().to(device)
    new_flat = np.zeros_like(raw4d.reshape(-1,P,P))
    A = mix4d.reshape(-1,P,P)
    with torch.no_grad():
        for i in range(1, scan_H-1):
            for j in range(1, scan_W-1):
                idx=[(i+di)*scan_W+(j+dj) for di,dj in NB_OFF]
                nb=torch.from_numpy(A[idx]).float().unsqueeze(0).to(device)
                new_flat[i*scan_W+j]=model(nb)[0,0].cpu().numpy()
    # borders unchanged
    for i in [0,scan_H-1]:
        for j in range(scan_W): new_flat[i*scan_W+j]=A[i*scan_W+j]
    for j in [0,scan_W-1]:
        for i in range(scan_H): new_flat[i*scan_W+j]=A[i*scan_W+j]

    den4d = new_flat.reshape(scan_H,scan_W,P,P)
    np.save(f'{OUT_DIR}/denoised_cycle_{c}.npy', den4d)
    torch.cuda.empty_cache(); gc.collect()

# ═════════════ save final outputs ═══════════════════════════════════
np.save(f'{OUT_DIR}/SrTiO3_denoised.npy', den4d)
torch.save(model.net.state_dict(), f'{OUT_DIR}/unet_denoiser.pt')
print('\n✔ Final denoised cube & weights saved.')

# ------------- bright‑field comparison -----------------------------
BF_raw  = raw4d.sum((2,3))
BF_deno = den4d.sum((2,3))
BF_high = hi4d.sum((2,3))

plt.figure(figsize=(11,3))
for k,(img,title) in enumerate([(BF_raw,'raw low‑dose'),
                                (BF_deno,'denoised (cycle 10)'),
                                (BF_high,'high‑dose')]):
    ax = plt.subplot(1,3,k+1); ax.imshow(img,cmap='inferno')
    ax.set_title(title); ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/denoised.png', dpi=300, bbox_inches='tight')
plt.show()
print('All outputs saved in', OUT_DIR)
