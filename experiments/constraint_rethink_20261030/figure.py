"""Standalone mathematical illustration, not an image editing result."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'outputs/deformation_example.png'
    if out.exists():raise RuntimeError('Immutable figure exists')
    points=np.array([[-1.,-.6],[1.,-.6],[1.,.6],[-1.,.6],[-1.,-.6]])
    nose=np.array([.6,.6])
    u=np.diag([2.,1.]);theta=.6
    r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,f,label,rot in zip(axes,[np.eye(2),u,r@u],['Source','Intrinsic symmetric stretch','Stretch + added rotation'],[0,0,theta]):
        xy=(f@points.T).T;vector=f@nose
        ax.plot(xy[:,0],xy[:,1],'-o',color='#24669c',markersize=4)
        ax.plot([0,vector[0]],[0,vector[1]],'-o',color='#bd3485',label='Two-point line')
        ax.set_aspect('equal');ax.set_xlim(-2.5,2.5);ax.set_ylim(-2.,2.)
        ax.axhline(0,color='gray',lw=.4);ax.axvline(0,color='gray',lw=.4)
        angle=np.arctan2(vector[1],vector[0])*180/np.pi
        ax.set_title(label+f'\nLine angle {angle:.1f} deg; polar R {rot*180/np.pi:.1f} deg',fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('A two-point angle can change under zero polar rotation.\nConstructed 2D example; polar rotation is not automatically semantic pose.',fontsize=10)
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


if __name__=='__main__':main()
