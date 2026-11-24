"""Fixed image transformations. No class labels or learned holdout statistics."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image,ImageOps
from legacy_reference import prepare_roi as legacy_roi

VARIANTS=['center_gray','adaptive_gray','adaptive_clahe']

def rotate_bound(rgb,angle):
    h,w=rgb.shape[:2];mat=cv2.getRotationMatrix2D((w/2,h/2),-angle,1)
    c,s=abs(mat[0,0]),abs(mat[0,1]);nw=int(np.ceil(h*s+w*c));nh=int(np.ceil(h*c+w*s))
    mat[0,2]+=nw/2-w/2;mat[1,2]+=nh/2-h/2
    return cv2.warpAffine(rgb,mat,(nw,nh),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def adaptive_roi(rgb):
    h,w=rgb.shape[:2];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),35,95)
    joined=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=2)
    joined=cv2.dilate(joined,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(joined,8)
    candidates=[]
    for i in range(1,n):
        x,y,bw,bh,area=stats[i];cx,cy=centers[i]
        if bh<.28*h or bw<.07*w or bw>.85*w or bh<1.15*bw:continue
        if not (.20*w<cx<.80*w) or area<.001*h*w:continue
        score=float(area)*(bh/h)/(1+4*abs(cx/w-.5))
        candidates.append((score,i))
    if not candidates:
        return rgb.copy(),{'roi_found':0,'roi_fallback':1,'axis_found':0,'axis_angle_deg':np.nan,
            'crop_x0':0,'crop_y0':0,'crop_x1':w,'crop_y1':h}
    _,idx=max(candidates);x,y,bw,bh,area=stats[idx]
    mx=int(.20*bw+8);my=int(.10*bh+8)
    x0=max(0,x-mx);x1=min(w,x+bw+mx);y0=max(0,y-my);y1=min(h,y+bh+my)
    crop=rgb[y0:y1,x0:x1].copy()
    ys,xs=np.nonzero(labels==idx);pts=np.c_[xs,ys].astype(float)
    centered=pts-pts.mean(0);cov=centered.T@centered/max(1,len(pts))
    values,vectors=np.linalg.eigh(cov);vec=vectors[:,-1]
    if vec[1]<0:vec=-vec
    angle=float(np.degrees(np.arctan2(vec[0],vec[1])))
    aligned=values[-1]/max(values[0],1e-8)>=3 and abs(angle)<=55
    if aligned:crop=rotate_bound(crop,angle)
    return crop,{'roi_found':1,'roi_fallback':0,'axis_found':int(aligned),'axis_angle_deg':angle if aligned else np.nan,
        'crop_x0':int(x0),'crop_y0':int(y0),'crop_x1':int(x1),'crop_y1':int(y1)}

def prepare(rgb,variant):
    if variant not in VARIANTS:raise ValueError(f'Unknown PCD variant: {variant}')
    if variant=='center_gray':
        roi,d=legacy_roi(rgb,'center_roi')
        info={k:v for k,v in d.items() if k not in {'axis_y','axis_x'}}
        info.update(roi_found=1,crop_y0=0,crop_y1=rgb.shape[0])
    else:roi,info=adaptive_roi(rgb)
    if variant.endswith('clahe'):
        lab=cv2.cvtColor(roi,cv2.COLOR_RGB2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(lab[:,:,0])
        roi=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return roi,info

def letterbox(rgb,size=224):
    im=ImageOps.pad(Image.fromarray(rgb),(size,size),method=Image.Resampling.LANCZOS,color=(128,128,128),centering=(.5,.5))
    return np.asarray(im).copy()
