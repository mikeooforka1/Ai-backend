"""Test doubles for InsightFace and Wav2Lip; never a production engine."""
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
import main
import lipsync

reference=np.array([1.,0.],np.float32)
selected=SimpleNamespace(bbox=np.array([40.,70.,100.,130.]),normed_embedding=reference,det_score=.99)
other=SimpleNamespace(bbox=np.array([230.,65.,285.,135.]),normed_embedding=np.array([0.,1.]),det_score=.99)
source=SimpleNamespace(bbox=np.array([10.,10.,65.,65.]),normed_embedding=reference,det_score=.99)
def faces(image):return [source] if image.shape[0]==80 else [other,selected]
def swap(image,*args,**kwargs):
    result=image.copy();result[70:130,40:100]=(30,170,35);return result

def initialize_face():
    main.FACE_APP=SimpleNamespace(get=faces)
    main.FACE_SWAPPER=SimpleNamespace(get=swap)
    main.FACE_ERROR=None

def initialize_lip():
    lipsync.MODEL=True;lipsync.ERROR=None

def predict(crops,windows):
    time.sleep(.06)
    return np.full((len(crops),96,96,3),(35,190,45),np.uint8)
main.initialize_face=initialize_face
lipsync.initialize=initialize_lip
lipsync.predict=predict
main.main()
