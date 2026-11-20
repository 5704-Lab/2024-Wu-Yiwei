"""Local native hooks, full token tensors, no processor replacement or B stream."""
import torch
from common import LAYERS


class NativeTap:
    def __init__(self,model):self.model=model

    def observe(self,y,eps,tau,condition):
        result={l:{} for l in LAYERS};handles=[]
        def norm_hook(l):
            def hook(module,args,output):result[l]['post_adaln']=output[0]
            return hook
        def input_hook(l):
            def hook(attn,args,kwargs):
                k=attn.to_k(kwargs['hidden_states'])
                b,n,d=k.shape
                k=k.reshape(b,n,attn.heads,d//attn.heads).transpose(1,2)
                if attn.norm_k is not None:k=attn.norm_k(k)
                result[l]['K']=k.transpose(1,2).flatten(2)
            return hook
        def projection_hook(l,facet):
            def hook(module,args):result[l][facet]=args[0]
            return hook
        try:
            for l in LAYERS:
                block=self.model.transformer_blocks[l]
                handles.extend([block.norm1.register_forward_hook(norm_hook(l)),
                    block.attn.register_forward_pre_hook(input_hook(l),with_kwargs=True),
                    block.attn.to_out[0].register_forward_pre_hook(projection_hook(l,'O')),
                    block.attn.to_add_out.register_forward_pre_hook(projection_hook(l,'C'))])
            pred=self.model(hidden_states=(1-tau)*y+tau*eps,timestep=torch.full((1,),tau*1000,device=y.device,dtype=y.dtype),
                encoder_hidden_states=condition['seq'].to(y),pooled_projections=condition['pool'].to(y),return_dict=False)[0]
        finally:
            for h in handles:h.remove()
        return result,pred
