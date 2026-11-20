"""Multiscale source-fixed sign-invariant gradient orientation observations."""
import math
import torch
import torch.nn.functional as F

class Geometry:
 def __init__(self,source,size=128,patch=8):
  self.size=size;self.patch=patch;self.kernels=[];self.weights=[];self.floors=[]
  for sigma in [.8,2.5]:
   radius=math.ceil(3*sigma);x=torch.arange(-radius,radius+1,device=source.device,dtype=source.dtype)
   k=torch.exp(-x*x/(2*sigma*sigma));k=k/k.sum();self.kernels.append((k[None,None,None,:],k[None,None,:,None]))
  with torch.no_grad():
   for gx,gy in self.gradients(source):
    energy=gx.square()+gy.square();floor=energy.flatten().median().clamp_min(1e-8)*.03
    self.floors.append(floor)
    self.weights.append((energy/(energy+floor*10)).sqrt())
   self.source=self(source).detach()

 def gradients(self,image):
  im=F.interpolate(image,(self.size,self.size),mode='bilinear',align_corners=False,antialias=True)
  im=(im*torch.tensor([.299,.587,.114],device=im.device,dtype=im.dtype)[None,:,None,None]).sum(1,keepdim=True)
  # Local photometric normalization, calibrated against legal illumination controls.
  kx,ky=self.kernels[1]
  rx=kx.shape[-1]//2;ry=ky.shape[-2]//2
  def local(z):
   z=F.conv2d(F.pad(z,(rx,rx,0,0),mode='reflect'),kx)
   return F.conv2d(F.pad(z,(0,0,ry,ry),mode='reflect'),ky)
  mean=local(im);variance=(local(im.square())-mean.square()).clamp_min(0.)
  im=(im-mean)/torch.sqrt(variance+.005**2)
  outputs=[]
  for kx,ky in self.kernels:
   rx=kx.shape[-1]//2;ry=ky.shape[-2]//2
   smooth=F.conv2d(F.pad(im,(rx,rx,0,0),mode='reflect'),kx)
   smooth=F.conv2d(F.pad(smooth,(0,0,ry,ry),mode='reflect'),ky)
   padded=F.pad(smooth,(1,1,1,1),mode='reflect')
   gx=(padded[:,:,1:-1,2:]-padded[:,:,1:-1,:-2])*.5
   gy=(padded[:,:,2:,1:-1]-padded[:,:,:-2,1:-1])*.5
   outputs.append((gx,gy))
  return outputs

 def __call__(self,image):
  descriptors=[]
  for j,(gx,gy) in enumerate(self.gradients(image)):
   energy=gx.square()+gy.square()
   tensor=torch.cat([gx.square()-gy.square(),2*gx*gy],1)/(energy+self.floors[j])
   descriptors.append(tensor*self.weights[j])
  z=torch.cat(descriptors,1);b,c,h,w=z.shape;p=self.patch
  return z.reshape(b,c,h//p,p,w//p,p).permute(0,2,4,1,3,5).reshape(b,(h//p)*(w//p),-1)

 def loss(self,obs,kappa=40.):
  e=.5*(obs-self.source).square().mean(-1)
  if kappa==0:return e.mean()
  return (torch.logsumexp(e*kappa,dim=-1)-math.log(e.shape[-1])).mean()/kappa

 def gn_weights(self,obs,kappa=40.):
  e=.5*(obs-self.source).square().mean(-1)
  return torch.softmax(e*kappa,dim=-1).detach() if kappa else torch.full_like(e,1/e.shape[-1])
