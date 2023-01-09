import av
import numpy as np
from av import AudioResampler


def get_duration_sec(file, cache=False):
    try:
        with open(file + '.dur', 'r') as f:
            duration = float(f.readline().strip('\n'))
        return duration
    except:
        container = av.open(file)
        audio = container.streams.get(audio=0)[0]
        duration = audio.duration * float(audio.time_base)
        if cache:
            with open(file + '.dur', 'w') as f:
                f.write(str(duration) + '\n')
        return duration


def load_audio(file, sr, offset, duration, resample=True, approx=False, time_base='samples', check_duration=True):
    if time_base == 'sec':
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    container = av.open(file)
    audio = container.streams.get(audio=0)[0]  # Only first audio stream
    audio_duration_sec = audio.duration * float(audio.time_base)
    if approx:
        raise NotImplementedError
        if offset + duration > audio_duration * sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration * sr - duration, offset - duration)
    else:
        if check_duration:
            assert offset + duration <= audio_duration_sec * sr, f'End {offset + duration} beyond duration {audio_duration_sec * sr}'
    if resample:
        resampler: AudioResampler = av.AudioResampler(format='fltp', layout='stereo', rate=sr)
    else:
        assert sr == audio.sample_rate
    offset_seconds = offset / sr
    offset_time_base = int(offset_seconds / float(audio.time_base))
    duration = int(duration)  # duration = int(duration * sr) # Use units of time_out ie 1/sr for returning
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset_time_base, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0):  # Only first audio stream
        if resample and sr != audio.sample_rate:
            frame.pts = None
            frame = resampler.resample(frame)[0]
        frame = frame.to_ndarray(format='fltp')  # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read:total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f'Expected {duration} frames, got {total_read}'
    return sig, sr
