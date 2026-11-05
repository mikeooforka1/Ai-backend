import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
import json
root=Path(sys.argv[2]);root.mkdir(exist_ok=True,parents=True)
if sys.argv[1]=='generate':
    rng=np.random.default_rng(7)
    texture=rng.integers(25,230,(60,60,3),dtype=np.uint8)
    writer=cv2.VideoWriter(str(root/'silent.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,240))
    occluded=cv2.VideoWriter(str(root/'occluded.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,240))
    cutwriter=cv2.VideoWriter(str(root/'cut.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(320,240))
    for i in range(40):
        frame=np.full((240,320,3),35,np.uint8)
        frame[70:130,40+i:100+i]=texture
        frame[65:135,230:285]=(180,75,30)
        writer.write(frame)
        hidden=frame.copy()
        if i>=20:hidden[70:130,40+i:100+i]=35
        occluded.write(hidden)
        cutwriter.write(frame if i<20 else np.full_like(frame, (255,255,255)))
    writer.release();cutwriter.release();occluded.release()
    subprocess.run(['ffmpeg','-nostdin','-y','-loglevel','error','-i',str(root/'silent.mp4'),'-f','lavfi','-i','sine=frequency=440:sample_rate=44100:duration=2','-map','0:v','-map','1:a','-c:v','copy','-c:a','aac','-shortest',str(root/'input.mp4')],check=True)
    source=np.zeros((80,80,4),np.uint8);source[:,:,:3]=(30,210,35);source[:,:,3]=255
    cv2.imwrite(str(root/'source.png'),source)
    (root/'invalid.mp4').write_text('not a video')
    subprocess.run(['ffmpeg','-nostdin','-y','-loglevel','error','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=1',str(root/'speech.wav')],check=True)
    (root/'invalid.wav').write_text('not audio')
elif sys.argv[1]=='verify':
    cap=cv2.VideoCapture(str(root/'output.mp4'));frames=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        frames.append(frame)
    cap.release()
    assert len(frames)==40, len(frames)
    for i,frame in enumerate(frames):
        selected=frame[88:112,58+i:82+i].mean(axis=(0,1))
        assert selected[1]>150 and selected[1]>selected[0]*2, (i,selected)
        other=frame[85:115,245:265].mean(axis=(0,1))
        assert other[0]>140 and other[1]<100, (i,other)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(root/'output.mp4')]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    assert any(s.get('codec_name')=='h264' for s in probe['streams'])
    print('40 moving-target frames verified; second object unchanged; audio and H.264 verified.')
