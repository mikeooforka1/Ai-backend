import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2
import importlib.util
try:
    from scipy.io import wavfile
except ImportError:
    wavfile = None
HAS_AUDIO = wavfile is not None and importlib.util.find_spec("librosa") is not None
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine'))
import main as engine
import lipsync

class LipSyncTests(unittest.TestCase):
    def test_identity_rejects_low_confidence_and_invalid_embeddings(self):
        from types import SimpleNamespace
        ref=np.array([1.,0.])
        for embedding,confidence in [(ref,.1),(np.array([np.nan,0.]),1),(np.ones(3),1)]:
            face=SimpleNamespace(normed_embedding=embedding,det_score=confidence)
            self.assertIsNone(engine.match_face(ref,[face]))
    def test_continuity_rejects_jumps_and_scale_changes(self):
        self.assertTrue(engine.continuous_face([10,10,50,50],[14,13,54,53]))
        self.assertFalse(engine.continuous_face([10,10,50,50],[250,150,290,190]))
        self.assertFalse(engine.continuous_face([10,10,50,50],[0,0,200,200]))
    def test_natural_blend_preserves_unchanged_pixels(self):
        original=np.full((100,100,3),80,np.uint8)
        swapped=original.copy();swapped[30:60,30:60]=100
        result=engine.natural_blend(original,swapped,[25,20,65,65])
        self.assertTrue(np.array_equal(result[:20],original[:20]))
        self.assertTrue(np.array_equal(result[20:25],original[20:25]))
        self.assertEqual(int(result[40,40,0]),93)
    def test_mouth_blend_preserves_eyes_and_other_people(self):
        frame=np.full((120,180,3),80,np.uint8)
        original=frame.copy()
        lipsync.blend_mouth(frame,np.full((96,96,3),150,np.uint8),[20,20,80,100])
        self.assertTrue(np.array_equal(frame[:,100:],original[:,100:]))
        self.assertTrue(np.array_equal(frame[20:52,20:80],original[20:52,20:80]))
        self.assertGreater(int(frame[78,50,0]),100)
    def test_mel_windows_follow_timestamps_and_do_not_wrap_video(self):
        mel=np.tile(np.arange(100,dtype=np.float32),(80,1))
        self.assertEqual(float(lipsync.mel_window(mel,0,20)[0,0]),0)
        self.assertEqual(float(lipsync.mel_window(mel,10,20)[0,0]),40)
        self.assertEqual(float(lipsync.mel_window(mel,99,20)[0,0]),84)
        self.assertEqual(lipsync.mel_window(mel[:,:3],0,25).shape,(80,16))
    @unittest.skipUnless(HAS_AUDIO, "Install optional lip-sync dependencies to test audio features")
    def test_audio_features_and_silent_audio_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'audio.wav'
            speech=(np.sin(2*np.pi*440*np.arange(16000)/16000)*10000).astype(np.int16)
            wavfile.write(path,16000,speech)
            mel=lipsync.mel_spectrogram(path)
            self.assertEqual(mel.shape[0],80)
            self.assertTrue(np.isfinite(mel).all())
            self.assertTrue(np.max(np.abs(mel))<=4)
            wavfile.write(path,16000,np.zeros(16000,np.int16))
            with self.assertRaisesRegex(ValueError,'silent'):lipsync.mel_spectrogram(path)
    def test_missing_model_has_actionable_error(self):
        with patch.object(lipsync,'MODEL',None):
            with self.assertRaisesRegex(ValueError,'model'):lipsync.predict([],[])
    @unittest.skipUnless(HAS_AUDIO, "Install optional lip-sync dependencies to test rendering")
    def test_cpu_render_skips_untracked_faces_and_streams_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.mp4';output=root/'output.mp4';wav=root/'speech.wav'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
            for _ in range(10):writer.write(np.full((120,160,3),80,np.uint8))
            writer.release()
            wavfile.write(wav,16000,(np.sin(2*np.pi*440*np.arange(16000)/16000)*10000).astype(np.int16))
            boxes=[[20,20,80,100] if i%2==0 else None for i in range(10)]
            updates=[]
            with patch.object(lipsync,'predict',lambda crops,windows:np.full((len(crops),96,96,3),160,np.uint8)):
                stats=lipsync.render(source,output,boxes,wav,10,lambda *a:updates.append(a))
            self.assertEqual(stats,dict(lipSyncFrames=5,lipSyncSkipped=5))
            self.assertEqual(updates[-1][0],10)
            cap=cv2.VideoCapture(str(output));self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),10);cap.release()
    def test_cancel_marker_is_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            marker=Path(directory)/'cancel';marker.write_text('cancel')
            with patch.object(engine,'CANCEL_PATH',marker):
                with self.assertRaisesRegex(ValueError,'cancelled'):engine.check_cancel()

if __name__=='__main__':unittest.main()
