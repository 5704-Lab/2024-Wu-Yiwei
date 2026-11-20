"""Fixed complex observations and bounded, source-anchored appearance mechanisms."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
OUT = HERE / 'outputs'
PAIRS = ['bear__00', 'brown_owl__00', 'bus__00', 'cat_crown__00']
sys.path.insert(0, str(ROOT / 'experiments/siglip_state_response'))
from run_precheck import CleanScorer


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def seed_setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = image.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.uint8(np.rint(arr.clip(0, 1)*255))).save(path)


def panel(columns, path, title='', size=256):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (len(columns)*size, size+65), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill='black')
    for i, (label, im) in enumerate(columns):
        if torch.is_tensor(im):
            arr = im.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(np.uint8(np.rint(arr.clip(0, 1)*255)))
        elif isinstance(im, (str, Path)):
            im = Image.open(im).convert('RGB')
        canvas.paste(im.resize((size, size)), (i*size, 55))
        draw.text((i*size+4, 25), label, fill='black')
    canvas.save(path)


def source_rgb(pair, device='cpu', dtype=torch.float32):
    data = torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt', weights_only=True)
    im = Image.open(ROOT/data['record']['source_image']).convert('RGB').resize((512,512))
    arr = np.asarray(im).copy()/255.
    return torch.tensor(arr, dtype=dtype, device=device).permute(2,0,1)[None], data


class Phase:
    def __init__(self, source, size=128, stride=4):
        self.size, self.pad, self.stride = size, 32, stride
        self.device, self.dtype = source.device, source.dtype
        self.color = torch.tensor([[.299,.587,.114], [.5,-.5,0.], [.25,.25,-.5]],
                                  device=self.device, dtype=self.dtype)
        n = size+2*self.pad
        f = torch.fft.fftfreq(n, device=self.device, dtype=self.dtype)
        fy, fx = torch.meshgrid(f, f, indexing='ij')
        kernels = []
        for freq in [1/16,1/8,1/4]:
            for angle in [0., math.pi/4, math.pi/2, 3*math.pi/4]:
                ax = fx*math.cos(angle)+fy*math.sin(angle)
                ay = -fx*math.sin(angle)+fy*math.cos(angle)
                width = freq*.45
                ker = torch.exp(-.5*((ax-freq)**2+ay**2)/width**2)
                # Exact zero DC; Hermitian symmetry is unnecessary for analysis.
                ker[0,0] = 0
                kernels.append(ker)
        self.kernels = torch.stack(kernels)
        self.source_small = self.small(source).detach()
        z = self.coefficients(source).detach()
        typical = z.abs().flatten(2).median(-1).values[...,None,None].clamp_min(1e-5)
        self.epsilon = .05*typical
        amp2 = z.abs().square()
        reliability = amp2/(amp2+(.3*typical).square())
        self.sqrt_weight = (reliability/reliability.sum()).sqrt()
        self.source_obs = self.embed(z).detach()

    def small(self, im):
        return F.interpolate(im, (self.size,self.size), mode='bilinear', align_corners=False, antialias=True)

    def coefficients(self, im):
        im = self.small(im)
        channels = torch.einsum('oc,bchw->bohw', self.color, im)
        padded = F.pad(channels, (self.pad,)*4, mode='reflect')
        z = torch.fft.ifft2(torch.fft.fft2(padded)[:,:,None]*self.kernels[None,None])
        z = z[...,self.pad:self.pad+self.size,self.pad:self.pad+self.size]
        # Merge color and band. Fixed lattice is not an edit-region mask.
        return z.flatten(1,2)[...,2::self.stride,2::self.stride]

    def embed(self, z):
        u = z / torch.sqrt(z.abs().square()+self.epsilon.square())
        return torch.view_as_real(u*self.sqrt_weight).flatten(1)

    def __call__(self, im):
        return self.embed(self.coefficients(im))

    def diagnostics(self, im):
        z = self.coefficients(im)
        return dict(low_amplitude_fraction=float((z.abs()<self.epsilon).float().mean()),
                    phase_source_distance=float((self.embed(z)-self.source_obs).norm()))


class Appearance:
    def __init__(self, source, order=2):
        self.source = source.detach()
        self.order = order
        h,w = source.shape[-2:]
        xx = (torch.arange(w,device=source.device,dtype=source.dtype)+.5)/w
        yy = (torch.arange(h,device=source.device,dtype=source.dtype)+.5)/h
        self.basis = torch.stack([torch.cos(math.pi*i*yy)[:,None]*torch.cos(math.pi*j*xx)[None,:]
                                  for i in range(order) for j in range(order)])
        self.count = order*order
        self.n = 6*self.count
        # L1 coefficient bounds imply pointwise gain/offset bounds.
        # gain field lies in [-1.4, 1.4], offset field in [-.15,.15].
        gain_bounds = [1.2]+[.2/max(self.count-1,1)]*(self.count-1)
        offset_bounds = [.12]+[.03/max(self.count-1,1)]*(self.count-1)
        self.bound = torch.tensor(gain_bounds*3+offset_bounds*3,dtype=source.dtype,device=source.device)

    def __call__(self, theta):
        theta = theta.reshape(-1,6,self.count)
        fields = torch.einsum('bcm,mhw->bchw',theta,self.basis)
        return self.source*fields[:,:3].exp()+fields[:,3:]

    def fit(self, phase, target, initial=None, maxiter=30):
        target = target.detach()
        x0 = np.zeros(self.n) if initial is None else initial.detach().cpu().double().numpy().copy()
        bounds = self.bound.cpu().double().numpy()
        calls = 0
        def fun(a):
            nonlocal calls
            calls += 1
            t = torch.tensor(a, device=self.source.device, dtype=self.source.dtype, requires_grad=True)
            r = phase(self(t))-target
            # Tiny numerical parameter prior is explicit, not data information.
            loss = .5*r.square().sum()+1e-7*(t/self.bound).square().mean()
            grad, = torch.autograd.grad(loss,t)
            return float(loss), grad.detach().cpu().double().numpy()
        opt = minimize(fun,x0,jac=True,method='L-BFGS-B',bounds=list(zip(-bounds,bounds)),
                       options=dict(maxiter=maxiter,ftol=1e-10,gtol=1e-7,maxls=20))
        theta = torch.tensor(opt.x,device=self.source.device,dtype=self.source.dtype)
        with torch.no_grad():
            ref = phase(self(theta)).detach()
        return theta,ref,dict(success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),calls=calls,
                              residual=float((ref-target).norm()),max_bound_ratio=float((theta/self.bound).abs().max()))

    def jacobian(self, phase, theta, eps=1e-3):
        # Symmetric small probes; no VAE needed for the reference family.
        eye = torch.eye(self.n, device=theta.device,dtype=theta.dtype)
        cols = []
        with torch.no_grad():
            for j in range(self.n):
                p=phase(self(theta+eps*eye[j]));m=phase(self(theta-eps*eye[j]))
                cols.append(((p-m)/(2*eps)).flatten())
        return torch.stack(cols,dim=1)


def orthogonal_basis(v, rtol=1e-3):
    u,s,_ = torch.linalg.svd(v.double(),full_matrices=False)
    keep = s > max(float(s.max())*rtol,1e-7)
    return u[:,keep].to(v.dtype),s.cpu().tolist(),int(keep.sum())


def warp(im, tx=0., ty=0., angle=0.):
    # tx,ty are pixels at this image's resolution, positive content displacement.
    h,w=im.shape[-2:]
    c,s=math.cos(angle),math.sin(angle)
    theta=torch.tensor([[[c,s,-2*tx/w],[-s,c,-2*ty/h]]],device=im.device,dtype=im.dtype)
    grid=F.affine_grid(theta,im.shape,align_corners=False)
    return F.grid_sample(im,grid,mode='bilinear',padding_mode='reflection',align_corners=False)


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
