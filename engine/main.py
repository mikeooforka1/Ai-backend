"""Private JSON-line engine, owned by Express. stdout is protocol only."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import time

import lipsync

import cv2
import numpy as np

FACE_APP = FACE_SWAPPER = None
FACE_ERROR = 'Face model not configured.'
CANCEL_PATH = None

def check_cancel():
    if CANCEL_PATH is not None and CANCEL_PATH.exists():
        raise ValueError('Processing cancelled by user.')

def emit(message):
    print(json.dumps(message, allow_nan=False), flush=True)

def initialize_face():
    global FACE_APP, FACE_SWAPPER, FACE_ERROR
    model = os.getenv('FACE_MODEL_PATH')
    root = os.getenv('FACE_MODEL_ROOT')
    if not model or not root:
        return
    try:
        if not Path(model).is_file():
            raise ValueError('FACE_MODEL_PATH does not exist.')
        name = os.getenv('FACE_ANALYSIS_NAME', 'buffalo_l')
        directory = Path(root) / 'models' / name
        if len(list(directory.glob('*.onnx'))) < 2:
            raise ValueError('Provide a local InsightFace analysis model pack under FACE_MODEL_ROOT/models/' + name)
        from insightface.app import FaceAnalysis
        from insightface.model_zoo import get_model
        # InsightFace/ONNX diagnostics must not corrupt stdout protocol.
        from contextlib import redirect_stdout
        with redirect_stdout(sys.stderr):
            app = FaceAnalysis(name=name, root=root, providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            swapper = get_model(model, providers=['CPUExecutionProvider'])
        if not swapper or not hasattr(swapper, 'get'):
            raise ValueError('Unsupported face swap model.')
        FACE_APP, FACE_SWAPPER = app, swapper
        FACE_ERROR = None
    except Exception as error:
        FACE_ERROR = str(error)
        print('Face mode disabled: ' + FACE_ERROR, file=sys.stderr)

def ffmpeg(args):
    result = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', *args], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise ValueError('FFmpeg failed: ' + result.stderr[-1200:])

def probe(video):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', video], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Video could not be decoded. Use a valid MP4, MOV or WebM file.')
    meta = json.loads(result.stdout)
    stream = next((s for s in meta.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not stream:
        raise ValueError('File contains no video stream.')
    w, h = int(stream['width']), int(stream['height'])
    duration = float(meta.get('format', {}).get('duration') or stream.get('duration') or 0)
    num, den = stream.get('avg_frame_rate', '0/1').split('/')
    fps = float(num) / float(den) if float(den) else 0
    if not (math.isfinite(duration) and 0 < duration <= float(os.getenv('MAX_DURATION_SECONDS', '120'))):
        raise ValueError('Video duration is missing or exceeds MAX_DURATION_SECONDS (default 120).')
    if w * h > int(os.getenv('MAX_PIXELS', '2073600')) or min(w, h) < 32:
        raise ValueError('Video resolution must be at least 32px and no larger than MAX_PIXELS (default 1080p).')
    if not (math.isfinite(fps) and 1 <= fps <= 60):
        raise ValueError('Supported frame rate is 1–60 FPS.')
    return {'width': w, 'height': h, 'duration': duration, 'fps': fps}

def normalize(video, destination, meta):
    # Bake rotation, square pixels and constant FPS into the exact video used for preview/tracking.
    ffmpeg(['-y', '-i', video, '-map', '0:v:0', '-an', '-vf', f"fps={meta['fps']},scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1", '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', destination])

def audio_probe(audio):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(audio)], capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError('Audio could not be decoded. Upload a valid WAV, MP3 or M4A file.')
    meta = json.loads(result.stdout)
    stream = next((s for s in meta.get('streams', []) if s.get('codec_type') == 'audio'), None)
    if not stream:
        raise ValueError('Uploaded audio contains no audio stream.')
    duration = float(meta.get('format', {}).get('duration') or stream.get('duration') or 0)
    if not math.isfinite(duration) or duration < .2 or duration > 120:
        raise ValueError('Audio must be between 0.2 and 120 seconds long.')
    return {'duration': duration, 'sampleRate': int(stream.get('sample_rate', 0))}

def inspect(video, image, dir, audio=None, **_):
    meta = probe(video)
    source = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if source is None or source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError('Replacement must be a decodable RGB/RGBA PNG or JPEG.')
    if source.shape[0] * source.shape[1] > 16000000:
        raise ValueError('Replacement image exceeds 16 megapixels.')
    if source.shape[2] == 4 and not np.any(source[:, :, 3]):
        raise ValueError('Replacement image is entirely transparent.')
    normalized = str(Path(dir) / 'normalized.mp4')
    normalize(video, normalized, meta)
    cap = cv2.VideoCapture(normalized)
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError('Unable to read first video frame.')
        meta.update(width=frame.shape[1], height=frame.shape[0], fps=cap.get(cv2.CAP_PROP_FPS))
        if not cv2.imwrite(str(Path(dir) / 'preview.jpg'), frame):
            raise ValueError('Unable to save selection preview.')
        if not cv2.imwrite(str(Path(dir) / 'source.png'), source):
            raise ValueError('Unable to save source preview.')
    finally:
        cap.release()
    return {**meta, 'hasAlpha': source.shape[2] == 4, 'audio': audio_probe(audio) if audio else None}

def box_pixels(selection, width, height):
    if not isinstance(selection, dict) or not all(isinstance(selection.get(k), (int, float)) and math.isfinite(selection[k]) for k in ('x','y','w','h')):
        raise ValueError('Invalid target selection.')
    x,y,w,h = (float(selection[k]) for k in ('x','y','w','h'))
    if x < 0 or y < 0 or w < .02 or h < .02 or x+w > 1.000001 or y+h > 1.000001:
        raise ValueError('Target selection must be inside the frame.')
    box = (round(x*width), round(y*height), round(w*width), round(h*height))
    if min(box[2:]) < 12:
        raise ValueError('Selected target must be at least 12 pixels wide and high.')
    return box

def composite(frame, source, box):
    x,y,w,h = map(int, box)
    if w <= 0 or h <= 0:
        return False
    x0,y0,x1,y1 = max(0,x),max(0,y),min(frame.shape[1],x+w),min(frame.shape[0],y+h)
    if x1 <= x0 or y1 <= y0:
        return False
    # Cover crop preserves source proportions instead of stretching.
    scale = max(w/source.shape[1], h/source.shape[0])
    sw,sh = max(w,round(source.shape[1]*scale)),max(h,round(source.shape[0]*scale))
    resized = cv2.resize(source, (sw,sh), interpolation=cv2.INTER_LINEAR)
    left,top = (sw-w)//2,(sh-h)//2
    layer = resized[top:top+h,left:left+w]
    alpha = layer[:,:,3].astype(np.float32)/255 if layer.shape[2]==4 else np.ones((h,w),np.float32)
    # Small feather avoids a hard rectangle edge. PNG alpha remains authoritative.
    feather = max(1, min(w,h)//40)
    edge = np.ones((h,w),np.float32)
    for k in range(feather):
        value = (k+1)/(feather+1)
        edge[k,:] = np.minimum(edge[k,:],value);edge[-k-1,:] = np.minimum(edge[-k-1,:],value)
        edge[:,k] = np.minimum(edge[:,k],value);edge[:,-k-1] = np.minimum(edge[:,-k-1],value)
    alpha = (alpha*edge)[y0-y:y1-y,x0-x:x1-x,None]
    roi = frame[y0:y1,x0:x1]
    foreground = layer[y0-y:y1-y,x0-x:x1-x,:3]
    roi[:] = np.clip(foreground*alpha + roi*(1-alpha),0,255).astype(np.uint8)
    return True

def face_get(image):
    from contextlib import redirect_stdout
    with redirect_stdout(sys.stderr):
        return FACE_APP.get(image)

def face_reference(first, source, box):
    if FACE_APP is None:
        raise ValueError(FACE_ERROR or 'Face mode unavailable.')
    candidates = face_get(first)
    x,y,w,h = box
    inside = [f for f in candidates if x <= (f.bbox[0]+f.bbox[2])/2 <= x+w and y <= (f.bbox[1]+f.bbox[3])/2 <= y+h]
    if len(inside) != 1:
        raise ValueError('Draw a box around exactly one visible face in the first frame.')
    sources = face_get(source[:,:,:3])
    if len(sources) != 1:
        raise ValueError('Face mode requires a replacement picture with exactly one visible face.')
    return inside[0].normed_embedding.copy(), sources[0]

def match_face(reference, faces):
    # Reject low-confidence and invalid embeddings before ranking identities.
    ranked = []
    for face in faces:
        embedding = np.asarray(face.normed_embedding)
        if embedding.shape != reference.shape or not np.isfinite(embedding).all() or float(getattr(face, 'det_score', 1)) < .5:
            continue
        score = float(np.dot(reference, embedding))
        if math.isfinite(score):
            ranked.append((score, face))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    if ranked and ranked[0][0] >= .5 and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= .08):
        return ranked[0][1]
    return None

def continuous_face(previous, current, gap=1):
    if previous is None:
        return True
    a, b = np.asarray(previous, dtype=float), np.asarray(current, dtype=float)
    if not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
        return False
    awh, bwh = a[2:] - a[:2], b[2:] - b[:2]
    ratio = np.prod(bwh) / max(1, np.prod(awh))
    distance = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
    return .3 <= ratio <= 3.3 and distance <= max(40, float(max(awh)) * (1 + min(gap, 10) * .15))

def natural_blend(original, swapped, bbox):
    # Keep the model's mask. Correct only changed pixels; never blend past frames.
    x0, y0, x1, y1 = lipsync.face_crop_box(bbox, original.shape[1], original.shape[0]) or [0, 0, 0, 0]
    roi, generated = original[y0:y1, x0:x1], swapped[y0:y1, x0:x1]
    if not roi.size:
        return swapped
    delta = np.max(np.abs(generated.astype(np.float32) - roi), axis=2)
    changed = delta > 2
    if np.count_nonzero(changed) < 10:
        return swapped
    correction = np.clip(np.median(roi[changed].astype(np.float32) - generated[changed], axis=0), -8, 8)
    weight = np.clip(delta / 24, 0, 1)[:, :, None]
    generated[:] = np.clip(generated.astype(np.float32) + correction * weight, 0, 255).astype(np.uint8)
    return swapped

def stage(jobId, name, progress, **details):
    check_cancel()
    emit({'event': 'progress', 'jobId': jobId, 'stage': name, 'progress': progress, **details})

def process(video, image, dir, jobId, mode, selection, audio=None, faceQuality="original", **_):
    if mode not in ('object','person','face','face_lipsync'):
        raise ValueError('Invalid replacement mode.')
    global CANCEL_PATH
    CANCEL_PATH = Path(dir) / (jobId + '.cancel')
    check_cancel()
    is_face = mode in ('face', 'face_lipsync')
    is_lip = mode == 'face_lipsync'
    if faceQuality not in ('original', 'natural'):
        raise ValueError('Unknown face blend profile.')
    if is_lip and lipsync.MODEL is None:
        raise ValueError(lipsync.ERROR)
    if is_lip and not audio:
        raise ValueError('Upload speech audio before rendering face swap + lip sync.')
    stage(jobId, 'validating', 1, detail='Checking media, selected identity and model readiness.')
    normalized = str(Path(dir) / 'normalized.mp4')
    source = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    cap = cv2.VideoCapture(normalized)
    writer = None
    temp = str(Path(dir) / (jobId + '.silent.mp4'))
    output = str(Path(dir) / (jobId + '.mp4'))
    lip_video = str(Path(dir) / (jobId + '.lips.mp4'))
    wav_path = str(Path(dir) / (jobId + '.speech.wav'))
    try:
        ok,first = cap.read()
        if not ok or source is None:
            raise ValueError('Input media is no longer available.')
        height,width = first.shape[:2]
        box = box_pixels(selection,width,height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        target_total = total
        audio_meta = None
        speech_mel = None
        if is_lip:
            if total / fps > float(os.getenv('LIPSYNC_MAX_SECONDS', '15')) or width * height > int(os.getenv('LIPSYNC_MAX_PIXELS', '921600')) or fps > 30.01:
                raise ValueError('CPU lip sync supports up to 15 seconds, 720p and 30 FPS by default. Trim/resize the clip first; start at 480p.')
            audio_meta = audio_probe(audio)
            target_total = min(total, int(audio_meta['duration'] * fps))
            if target_total < 1:
                raise ValueError('Audio is too short for this video frame rate.')
            stage(jobId, 'validating', 2, detail='Decoding speech and checking audio features before swapping frames.')
            ffmpeg(['-y', '-i', audio, '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-t', str(target_total/fps), wav_path])
            speech_mel = lipsync.mel_spectrogram(wav_path)
        writer = cv2.VideoWriter(temp,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
        if not writer.isOpened():
            raise ValueError('Video encoder unavailable.')
        tracker = None
        reference = source_face = None
        if is_face:
            reference,source_face = face_reference(first,source,box)
        else:
            tracker = cv2.TrackerCSRT_create()
            tracker.init(first,box)
        count = replaced = skipped = 0
        active = True
        lost_at = None
        previous_hist = None
        target_hist = None
        if tracker is not None:
            x,y,w,h = box
            target_hist = cv2.calcHist([cv2.cvtColor(first[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
            cv2.normalize(target_hist,target_hist)
        frame = first
        last_face = None
        last_seen = 0
        face_boxes = []
        face_cut = False
        stage(jobId, 'face_swap' if is_face else 'tracking', 3, framesTotal=target_total, detail='Matching the selected identity.' if is_face else 'Tracking the selected image region.')
        frame_started = time.monotonic()
        while True:
            check_cancel()
            count += 1
            changed = False
            if is_face:
                hist = cv2.calcHist([cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
                cv2.normalize(hist, hist)
                if previous_hist is not None and cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > .7:
                    face_cut = True
                    if lost_at is None:
                        lost_at = count
                previous_hist = hist
                faces = face_get(frame) if not face_cut else []
                target = match_face(reference, faces)
                if target is not None and (not continuous_face(last_face, target.bbox, count-last_seen) or min(target.bbox[2:] - target.bbox[:2]) < 24):
                    target = None
                if last_face is not None and count-last_seen > max(1, round(fps * .75)):
                    target = None
                # Fixed identity reference; do not gradually drift to another person.
                if target is not None:
                    from contextlib import redirect_stdout
                    with redirect_stdout(sys.stderr):
                        original = frame.copy() if faceQuality == 'natural' else None
                        frame = FACE_SWAPPER.get(frame,target,source_face,paste_back=True)
                    if frame is None or frame.shape != first.shape or frame.dtype != np.uint8:
                        raise ValueError('Face model returned an invalid image.')
                    if original is not None:
                        frame = natural_blend(original, frame, target.bbox)
                    last_face, last_seen = target.bbox.copy(), count
                    changed = True
                if is_lip:
                    face_boxes.append(lipsync.face_crop_box(target.bbox, width, height) if changed else None)
            else:
                hist = cv2.calcHist([cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
                cv2.normalize(hist,hist)
                # Stop replacing after a scene cut or lost tracking rather than jump targets.
                if previous_hist is not None and cv2.compareHist(previous_hist,hist,cv2.HISTCMP_BHATTACHARYYA) > .7:
                    active = False
                previous_hist = hist
                if active and count > 1:
                    ok, candidate = tracker.update(frame)
                    if not ok:
                        active = False
                    else:
                        nx,ny,nw,nh = candidate
                        bx,by,bw,bh = box
                        if nw<=0 or nh<=0 or nw*nh>width*height or abs(nx-bx)>max(80,bw*1.5) or abs(ny-by)>max(80,bh*1.5):
                            active = False
                        else:
                            box = candidate
                if active and count > 1:
                    x,y,w,h = map(int,box)
                    roi = frame[max(0,y):min(height,y+h),max(0,x):min(width,x+w)]
                    if roi.size == 0:
                        active = False
                    else:
                        appearance = cv2.calcHist([cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)],[0,1],None,[24,24],[0,180,0,256])
                        cv2.normalize(appearance,appearance)
                        if cv2.compareHist(target_hist,appearance,cv2.HISTCMP_BHATTACHARYYA) > .65:
                            active = False
                if active:
                    changed = composite(frame,source,box)
                elif lost_at is None:
                    lost_at = count
            replaced += int(changed)
            skipped += int(not changed)
            writer.write(frame)
            if count==1 or count%10==0:
                stage(jobId, 'face_swap' if is_face else 'tracking', min(50 if is_lip else 88, 3+round(count/target_total*(47 if is_lip else 85))), framesProcessed=count, framesTotal=target_total, framesReplaced=replaced, framesSkipped=skipped, stageFps=round(count/max(.001,time.monotonic()-frame_started),2))
            if count >= target_total:
                break
            ok,frame = cap.read()
            if not ok:
                break
        writer.release();writer=None;cap.release()
        if not replaced:
            raise ValueError('No frames could be replaced. Choose a clearer target and source image.')
        lip_stats = {}
        encode_video, encode_audio = temp, video
        warnings = ([f'Tracking stopped at frame {lost_at}; later frames were left unchanged.'] if lost_at else []) + ([f'{skipped} frames were not replaced. Review the output.'] if skipped else [])
        if is_lip:
            stage(jobId, 'audio', 52, detail='Preparing uploaded speech at 16 kHz; output uses the shorter of video and audio.')
            stage(jobId, 'lip_sync', 55, detail='Generating the mouth movements on CPU; only tracked faces are edited.')
            started = time.monotonic()
            def lip_progress(done, total, edited, missed):
                stage(jobId, 'lip_sync', min(90,55+round(done/total*35)), lipSyncProcessed=done, lipSyncTotal=total, lipSyncFrames=edited, lipSyncSkipped=missed, stageFps=round(done/max(.001,time.monotonic()-started),2))
            lip_stats = lipsync.render(temp, lip_video, face_boxes, wav_path, fps, lip_progress, mel=speech_mel)
            encode_video, encode_audio = lip_video, audio
            if audio_meta['duration'] > count/fps + .1:
                warnings.append('Uploaded audio was trimmed to the video duration; the video was not looped.')
            if count < total:
                warnings.append('Output ends with the uploaded audio; remaining video frames were not included.')
            if lip_stats['lipSyncSkipped']:
                warnings.append(f"Lip sync skipped {lip_stats['lipSyncSkipped']} unclear or untracked faces; those frames retain the face-swap result.")
        stage(jobId, 'encoding', 93, detail='Encoding H.264 video and AAC audio for browser playback.')
        ffmpeg(['-y','-i',encode_video,'-i',encode_audio,'-map','0:v:0','-map','1:a:0?' if not is_lip else '1:a:0','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-t',str(count/fps),'-movflags','+faststart',output])
        stage(jobId, 'verifying', 98, detail='Verifying output duration, resolution and audio stream.')
        encoded = probe(output)
        if encoded['width'] != width or encoded['height'] != height or abs(encoded['duration'] - count/fps) > max(.25,2/fps):
            raise ValueError('Encoded output failed timing or resolution verification.')
        if is_lip:
            audio_probe(output)
        return {'framesTotal':count,'framesReplaced':replaced,'framesSkipped':skipped,'trackingLostAt':lost_at,'warnings':warnings,'outputDuration':count/fps,'faceQuality':faceQuality, **lip_stats}

    except Exception:
        Path(output).unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        CANCEL_PATH = None
        for file in (temp, lip_video, wav_path):
            Path(file).unlink(missing_ok=True)

def main():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('Install FFmpeg and ffprobe on PATH before starting the backend.')
    initialize_face()
    lipsync.initialize()
    emit({'event':'ready','capabilities':{'object':True,'person':True,'face':FACE_APP is not None,'faceError':FACE_ERROR,'face_lipsync':FACE_APP is not None and lipsync.MODEL is not None,'lipSyncError':lipsync.ERROR if lipsync.MODEL is None else FACE_ERROR,'device':'cpu'}})
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            command = request.pop('command')
            ident = request.pop('id')
            handler = {'inspect':inspect,'process':process}.get(command)
            if handler is None:
                raise ValueError('Unknown engine command.')
            emit({'id':ident,'result':handler(**request)})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({'id':request.get('id',locals().get('ident')), 'error':str(error)})

if __name__ == '__main__':
    main()
