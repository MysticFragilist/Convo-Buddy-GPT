import pyaudio
import configparser

FILE_NAME = "config.cfg"

openAIKey = input('OpenAI API key: ')

p = pyaudio.PyAudio()

print("Microphone indexes with their names:")
for i in range(p.get_device_count()):
    print(str(i) + "-  " + p.get_device_info_by_index(i)['name'])

config = configparser.ConfigParser()

config.set('DEFAULT', 'openaikey', openAIKey)
config.set('DEFAULT', 'microphoneindex', input('Microphone index: '))

with open(FILE_NAME, 'w') as configfile:
    config.write(configfile)