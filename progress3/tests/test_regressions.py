import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from features import WIDTH,FEATURE_COLUMNS,glcm,reference_apply,reference_fit,select_features
from common import CLASSES,read_split
from preprocessing import letterbox

class FeatureRegressions(unittest.TestCase):
    def test_missing_profile_not_zero(self):
        frame=pd.DataFrame({c:[np.nan] for c in WIDTH})
        fixed=reference_apply(frame,{c:.5 for c in WIDTH})
        self.assertTrue(np.isnan(fixed.loc[0,'profile_dev_fraction_over_03']))

    def test_fraction_uses_valid_measurements(self):
        frame=pd.DataFrame({c:[np.nan] for c in WIDTH});frame.loc[0,WIDTH[0]]=.6
        fixed=reference_apply(frame,{c:.5 for c in WIDTH})
        self.assertAlmostEqual(fixed.loc[0,'profile_dev_fraction_over_03'],1)

    def test_numeric_labels_and_identifiers_excluded(self):
        frame=pd.DataFrame({c:[0.] for c in FEATURE_COLUMNS})
        frame['class_id']=2;frame['photo_number']=4811;frame['source_size_bytes']=123
        selected=select_features(frame)
        self.assertEqual(list(selected),FEATURE_COLUMNS)
        self.assertNotIn('class_id',selected)

    def test_glcm_energy_is_sqrt_asm(self):
        image=np.tile(np.array([0,255],dtype=np.uint8),(12,6))
        f=glcm(image,np.ones(image.shape,bool))
        self.assertAlmostEqual(f['glcm_asm_h'],.5)
        self.assertAlmostEqual(f['glcm_energy_h'],np.sqrt(.5))
        self.assertAlmostEqual(f['glcm_homogeneity_h'],1/226)

    def test_glcm_distance_changes_pixel_pairs(self):
        image=np.tile(np.array([0,255],dtype=np.uint8),(12,6));mask=np.ones(image.shape,bool)
        d1=glcm(image,mask,1);d2=glcm(image,mask,2)
        self.assertAlmostEqual(d1['glcm_contrast_h'],1.)
        self.assertAlmostEqual(d2['glcm_contrast_h'],0.)
        with self.assertRaises(ValueError):glcm(image,mask,0)

    def test_padding_preserves_dimensions(self):
        out=letterbox(np.full((400,100,3),255,dtype=np.uint8))
        self.assertEqual(out.shape,(224,224,3))
        self.assertTrue((out[:,0,:]==128).all())
        self.assertTrue((out[:,112,:]==255).all())

    def test_reference_uses_only_normal_train(self):
        f=pd.DataFrame({c:[.5,99.,.7] for c in WIDTH})
        metadata=pd.DataFrame({'class_name':['normal','penyok','normal']})
        r=reference_fit(f,metadata)
        self.assertAlmostEqual(r[WIDTH[0]],.6)

    def test_class_mapping_and_augmentation_plan(self):
        from deep_learning import augmentation_plan
        metadata=pd.DataFrame({'image_id':['a','b'],'class_id':[0,1]})
        a=augmentation_plan(metadata,3);b=augmentation_plan(metadata,3)
        self.assertEqual(a,b);self.assertEqual(len(a),8)
        self.assertEqual(sum(x['augmentation_index']==0 for x in a),2)
        self.assertEqual(set(x['image_id'] for x in a),{'a','b'})

    def test_device_and_head_forward(self):
        import torch
        from deep_learning import choose_device,new_head
        head=new_head(.2).eval()
        self.assertEqual(tuple(head(torch.zeros(2,960)).shape),(2,4))
        self.assertEqual(str(choose_device('cpu')),'cpu')

    def test_vectorized_importance_scores_match_sklearn(self):
        from explain import batched_macro_f1
        from sklearn.metrics import f1_score
        truth=np.array([0,0,1,1,2,2,3,3])
        predictions=np.array([[0,0,1,1,2,2,3,3],[0,1,2,3,0,1,2,3]])
        actual=batched_macro_f1(predictions,truth)
        expected=[f1_score(truth,p,labels=[0,1,2,3],average='macro',zero_division=0) for p in predictions]
        np.testing.assert_allclose(actual,expected)

    def test_channel_importance_zero_weight_is_zero(self):
        from explain import channel_importance
        x=np.array([[1.,3.],[2.,4.],[3.,5.],[4.,6.]])
        w=np.array([[1.,0.],[-1.,0.],[0.,0.],[0.,0.]])
        result=channel_importance(x,w,np.zeros(4),np.array([0,1,2,3]),repeats=5)
        self.assertAlmostEqual(result.loc[1,'importance_mean'],0.)

if __name__=='__main__':unittest.main()
