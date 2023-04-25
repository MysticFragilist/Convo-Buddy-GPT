import pyaudio
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import sounddevice
import wave
import time
import hashlib
import numpy as np

CHANNELS = 1
FRAME_RATE = 16000
RECORD_SECONDS = 10
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_SIZE = 2

# index 1 is the microphone

# TODO Find your microphone index using this snippet and put it in config.cfg
# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     print(p.get_device_info_by_index(i))


def record_microphone(messages, recordings, mic_index, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=FRAME_RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=chunk)
    frames = []

    while not messages.empty():
        data = stream.read(chunk)
        frames.append(data)

        if len(frames) >= (FRAME_RATE * RECORD_SECONDS) / chunk:
            recordings.put(frames)
            frames = []

    stream.stop_stream()
    stream.close()
    p.terminate()

# download and load all models
preload_models(
    text_use_small=True,
    coarse_use_small=True,
    fine_use_gpu=True,
    fine_use_small=True,
)


def play_audio(text):
    audio_array = generate_audio(text=text, silent=True, history_prompt="en_speaker_1")
    audio_array *= 32767 / 1.414
    hash_object = hashlib.md5(text.encode())
    write_wav("cache/" + hash_object.hexdigest() + ".wav", SAMPLE_RATE, audio_array.astype(np.int16))

    with wave.open("cache/" + hash_object.hexdigest()+ ".wav", 'rb') as wf:
        # Define callback for playback (1)
        def callback(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)
            # If len(data) is less than requested frame_count, PyAudio automatically
            # assumes the stream is finished, and the stream stops.
            return (data, pyaudio.paContinue)

        # Instantiate PyAudio and initialize PortAudio system resources (2)
        p = pyaudio.PyAudio()

        # Open stream using callback (3)
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback)

        # Wait for stream to finish (4)
        while stream.is_active():
            time.sleep(0.1)

        # Close the stream (5)
        stream.close()

        # Release PortAudio system resources (6)
        p.terminate()