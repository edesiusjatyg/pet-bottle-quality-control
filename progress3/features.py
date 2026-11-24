"""Explicit feature schema; metadata and labels never become predictors."""
from __future__ import annotations
import cv2,numpy as np,pandas as pd
from preprocessing import prepare

PROFILE_Y=np.array([.28,.34,.40,.46,.52,.58,.64,.70,.76,.82])
WIDTH=[f'profile_width_{i}' for i in range(10)]
LEFT=[f'profile_left_{i}' for i in range(10)]
RIGHT=[f'profile_right_{i}' for i in range(10)]
SHAPE=['edge_density','closed_edge_density','profile_valid_fraction','profile_width_median','profile_width_iqr']+WIDTH+LEFT+RIGHT
LABEL=['label_found','label_coverage','label_area_ratio','label_angle_relative_deg','label_angle_valid']
TEXTURE=[f'glcm_{stat}_{direction}' for direction in ['h','v'] for stat in ['contrast','homogeneity','asm','energy']]+['gray_mean','gray_std']
DERIVED=[f'normal_width_deviation_{i}' for i in range(10)]+['profile_dev_mean_abs','profile_dev_max_abs','profile_dev_fraction_over_03']
FEATURE_COLUMNS=SHAPE+LABEL+TEXTURE+DERIVED
assert len(FEATURE_COLUMNS)==len(set(FEATURE_COLUMNS))

def glcm(gray,mask,distance=1):
    if not isinstance(distance,(int,np.integer)) or distance<1:raise ValueError('GLCM distance d must be a positive integer')
    q=np.minimum(gray.astype(np.uint16)*16//256,15).astype(np.int32);out={}
    for dy,dx,d in [(0,distance,'h'),(distance,0,'v')]:
        if dy>=gray.shape[0] or dx>=gray.shape[1]:
            out.update({f'glcm_{s}_{d}':np.nan for s in ['contrast','homogeneity','asm','energy']});continue
        a=q[:q.shape[0]-dy,:q.shape[1]-dx];b=q[dy:,dx:]
        valid=mask[:q.shape[0]-dy,:q.shape[1]-dx]&mask[dy:,dx:]
        if valid.sum()<20:
            out.update({f'glcm_{s}_{d}':np.nan for s in ['contrast','homogeneity','asm','energy']});continue
        p=np.bincount((16*a[valid]+b[valid]),minlength=256).reshape(16,16).astype(float)
        p+=p.T.copy();p/=p.sum();i,j=np.indices((16,16));asm=float((p*p).sum())
        out.update({f'glcm_contrast_{d}':float((p*(i-j)**2).sum()/225),
            f'glcm_homogeneity_{d}':float((p/(1+(i-j)**2)).sum()),f'glcm_asm_{d}':asm,f'glcm_energy_{d}':float(np.sqrt(asm))})
    return out

def extract(rgb,variant,glcm_distance=1):
    roi,info=prepare(rgb,variant);h,w=roi.shape[:2];gray=cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
    edge=cv2.Canny(gray,35,95);closed=cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    f={'edge_density':float((edge>0).mean()),'closed_edge_density':float((closed>0).mean())}
    widths=[]
    for i,yy in enumerate(PROFILE_Y):
        y=int(yy*h);band=edge[max(0,y-3):min(h,y+4)]
        l0,l1=int(.08*w),int(.48*w);r0,r1=int(.52*w),int(.92*w)
        lp=band[:,l0:l1].sum(0);rp=band[:,r0:r1].sum(0)
        valid=bool(lp.size and rp.size)
        if valid:
            lx=l0+int(lp.argmax());rx=r0+int(rp.argmax())
            valid=min(lp.max(),rp.max())>=510 and .12<=(rx-lx)/w<=.9
        for col,value in [(WIDTH[i],(rx-lx)/w if valid else np.nan),(LEFT[i],lx/w if valid else np.nan),(RIGHT[i],rx/w if valid else np.nan)]:f[col]=float(value)
        widths.append(f[WIDTH[i]])
    vals=np.array(widths);valid_vals=vals[np.isfinite(vals)]
    f.update(profile_valid_fraction=float(np.isfinite(vals).mean()),profile_width_median=float(np.median(valid_vals)) if len(valid_vals) else np.nan,
        profile_width_iqr=float(np.ptp(np.percentile(valid_vals,[25,75]))) if len(valid_vals)>1 else np.nan)
    hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
    # Relative contrast helps reduce broad, white backgrounds; mask remains a heuristic.
    white=((hsv[:,:,1]<105)&(hsv[:,:,2]>130)).astype(np.uint8)*255
    region=np.zeros((h,w),np.uint8);region[int(.22*h):int(.78*h),int(.12*w):int(.88*w)]=255
    white=cv2.bitwise_and(white,region)
    white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    label_candidates=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c);area=cv2.contourArea(c)
        if .008<=area/(h*w)<=.30 and bw>.16*w and bh>.025*h and bw>1.1*bh:
            label_candidates.append((area*(1-abs((y+bh/2)/h-.48)),c))
    found=bool(label_candidates);label_mask=np.zeros((h,w),np.uint8);angle=np.nan
    if found:
        _,c=max(label_candidates,key=lambda x:x[0]);cv2.drawContours(label_mask,[c],-1,255,-1)
        points=c.reshape(-1,2).astype(float);rect=cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
        sides=np.roll(rect,-1,axis=0)-rect;v=sides[np.argmax(np.linalg.norm(sides,axis=1))]
        angle=float(np.degrees(np.arctan2(v[1],v[0])));angle=(angle+90)%180-90
        if not info.get('axis_found'):angle=np.nan
    f.update(label_found=float(found),label_coverage=float((white>0).sum()/max(1,(region>0).sum())),
        label_area_ratio=float((label_mask>0).mean()) if found else np.nan,
        label_angle_relative_deg=angle,label_angle_valid=float(np.isfinite(angle)))
    body=np.zeros((h,w),bool);body[int(.20*h):int(.90*h),int(.15*w):int(.85*w)]=True
    body[label_mask>0]=False
    f.update(glcm(gray,body,glcm_distance));pixels=gray[body]
    f.update(gray_mean=float(pixels.mean()) if len(pixels) else np.nan,gray_std=float(pixels.std()) if len(pixels) else np.nan)
    # Return only fixed measurements. Identifiers remain in a separate dataframe.
    return f,info,roi

def reference_fit(train_features,train_metadata):
    normals=train_features.loc[train_metadata['class_name'].to_numpy()=='normal',WIDTH]
    if len(normals)==0:raise ValueError('No normal training images')
    return {c:float(normals[c].median()) if normals[c].notna().any() else None for c in WIDTH}

def reference_apply(frame,reference):
    frame=frame.copy();deviations=[]
    for i,c in enumerate(WIDTH):
        value=reference.get(c)
        delta=frame[c]-value if value is not None else pd.Series(np.nan,index=frame.index)
        frame[DERIVED[i]]=delta;deviations.append(delta)
    d=pd.concat(deviations,axis=1).abs();valid=d.notna().sum(axis=1)
    frame['profile_dev_mean_abs']=d.mean(axis=1);frame['profile_dev_max_abs']=d.max(axis=1)
    # NaN is never counted as a successful zero deviation.
    frame['profile_dev_fraction_over_03']=(d>.03).sum(axis=1).div(valid.replace(0,np.nan))
    return frame.reindex(columns=FEATURE_COLUMNS)

def select_features(frame):
    missing=set(FEATURE_COLUMNS)-set(frame)
    if missing:raise ValueError(f'Missing measurements: {sorted(missing)}')
    return frame.loc[:,FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan)
