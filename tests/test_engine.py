import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
import main as engine

class EngineTests(unittest.TestCase):
    def test_alpha_does_not_change_transparent_pixels_or_outside_box(self):
        frame=np.full((100,100,3),60,np.uint8)
        image=np.full((30,30,4),255,np.uint8);image[:10,:,3]=0
        engine.composite(frame,image,(20,20,30,30))
        self.assertTrue(np.all(frame[:20]==60))
        self.assertTrue(np.all(frame[20:30,20:50]==60))
        self.assertTrue(np.all(frame[35:40,35:40]==255))
    def test_edge_boxes_clip_safely(self):
        frame=np.zeros((30,30,3),np.uint8);image=np.full((20,20,3),255,np.uint8)
        self.assertTrue(engine.composite(frame,image,(-5,-5,20,20)))
        self.assertFalse(engine.composite(frame,image,(40,40,20,20)))
    def test_invalid_selection_is_rejected(self):
        for selection in [None,{},dict(x=0,y=0,w=float('inf'),h=.3),dict(x=.9,y=0,w=.3,h=.3),dict(x=0,y=0,w=.02,h=.02)]:
            with self.assertRaises(ValueError):engine.box_pixels(selection,100,100)
    def test_face_identity_gate_preserves_selected_identity(self):
        ref=np.array([1.,0.]); correct=SimpleNamespace(normed_embedding=ref);other=SimpleNamespace(normed_embedding=np.array([0.,1.]))
        self.assertIs(engine.match_face(ref,[other,correct]),correct)
        self.assertIs(engine.match_face(ref,[correct,other]),correct)
        self.assertIsNone(engine.match_face(ref,[other]))
    def test_ambiguous_or_missing_faces_are_skipped(self):
        ref=np.array([1.,0.]);a=SimpleNamespace(normed_embedding=np.array([.8,.6]));b=SimpleNamespace(normed_embedding=np.array([.79,.613]))
        self.assertIsNone(engine.match_face(ref,[a,b]))
        self.assertIsNone(engine.match_face(ref,[]))
    def test_face_box_selects_one_target_and_one_source(self):
        selected=SimpleNamespace(bbox=np.array([20,20,40,40]),normed_embedding=np.array([1.,0.]))
        other=SimpleNamespace(bbox=np.array([60,60,80,80]),normed_embedding=np.array([0.,1.]))
        source=SimpleNamespace(bbox=np.array([0,0,10,10]),normed_embedding=np.array([.8,.6]))
        responses=iter([[other,selected],[source]])
        with patch.object(engine,'FACE_APP',SimpleNamespace(get=lambda image:next(responses))):
            reference,identity=engine.face_reference(np.zeros((100,100,3),np.uint8),np.zeros((100,100,3),np.uint8),(15,15,30,30))
        self.assertTrue(np.array_equal(reference,selected.normed_embedding))
        self.assertIs(identity,source)
    def test_multiple_selected_faces_are_rejected(self):
        faces=[SimpleNamespace(bbox=np.array([10,10,30,30])),SimpleNamespace(bbox=np.array([20,20,40,40]))]
        with patch.object(engine,'FACE_APP',SimpleNamespace(get=lambda image:faces)):
            with self.assertRaises(ValueError):engine.face_reference(np.zeros((100,100,3),np.uint8),np.zeros((100,100,3),np.uint8),(0,0,60,60))
    def test_unconfigured_face_mode_fails_explicitly(self):
        with self.assertRaises(ValueError):engine.face_reference(np.zeros((10,10,3),np.uint8),np.zeros((10,10,3),np.uint8),(0,0,10,10))

if __name__=='__main__':unittest.main()
