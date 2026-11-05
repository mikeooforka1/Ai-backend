"""Optional REAL CPU lip model test. Requires official weights configured in env.
No face detector or face swap models are mocked/claimed by this isolated test.
"""
import os
from pathlib import Path
import sys
import tempfile
import time
import json
import cv2
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine'))
import lipsync

def main():
    lipsync.initialize()
    if lipsync.MODEL is None:
        raise RuntimeError(lipsync.ERROR)
    source_image = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if source_image is None:
        raise ValueError('Pass a portrait image path for the real inference test.')
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);video=root/'input.mp4';output=root/'lips.mp4';speech=root/'speech.wav'
        image=cv2.resize(source_image,(512,512))
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),10,(512,512))
        for _ in range(10):writer.write(image)
        writer.release()
        wavfile.write(speech,16000,(np.sin(2*np.pi*220*np.arange(16000)/16000)*10000).astype(np.int16))
        # NASA/scikit-image astronaut portrait face coordinates, known in the fixture.
        boxes=[[175,65,275,180]]*10
        started=time.monotonic()
        stats=lipsync.render(video,output,boxes,speech,10,lambda *args:None)
        assert stats=={'lipSyncFrames':10,'lipSyncSkipped':0},stats
        original=cv2.VideoCapture(str(video));edited=cv2.VideoCapture(str(output))
        changes=[]
        for i in range(10):
            ok,a=original.read();good,b=edited.read();assert ok and good
            mask=np.ones((512,512),bool);mask[65:180,175:275]=False
            # Compare decoded frames, allowing MPEG4 re-encoding drift.
            assert float(np.mean(np.abs(a[mask].astype(float)-b[mask])))<3
            changes.append(float(np.mean(np.abs(a[120:170,185:265].astype(float)-b[120:170,185:265]))))
        original.release();edited.release()
        assert max(changes)>2,changes
        report={'model':'official Wav2Lip GAN TorchScript', 'device':'cpu', 'frames':10, 'elapsedSeconds':round(time.monotonic()-started,2), 'mouthRegionMeanChange':round(float(np.mean(changes)),2),'outsideFacePreserved':True,'note':'Tests real inference and regional isolation with a portrait and tone, not perceptual speech-sync quality or full face-swap accuracy.'}
        print(json.dumps(report))
        target=os.getenv('NEURAL_REPORT_PATH')
        if target:Path(target).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
