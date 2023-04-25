from threading import Thread
from queue import Queue
import sr_handler
import audio_handler
import config_handler

messages = Queue()
recordings = Queue()
texts = Queue()

def start_recording():
    if not messages.empty():
        print("Recording already started")
        return
    
    messages.put(True)
    print("Recording started")

    record = Thread(target=audio_handler.record_microphone, args=(messages, recordings, int(config_handler.getConfig('MicrophoneIndex'))))
    record.start()

    transcribe = Thread(target=sr_handler.speech_recognition, args=(messages, recordings))
    transcribe.start()


def stop_recording():
    if messages.empty():
        print("Recording already stopped")
        return
    messages.get()
    print("Recording stopped")


while True:
    print("Write 'start' to start recording or 'stop' to stop recording:")
    input1 = input()
    if input1 == "start":
        start_recording()
    elif input1 == "stop":
        stop_recording()
    elif input1 == "exit":
        stop_recording()
        break