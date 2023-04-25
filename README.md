# Convo buddy
An AI conversational buddy. Voice to text, process by GPT then response to voice friend

## Requirements

Cuda 11.7

Python: ^3.7 (ran with python 3.11)

```
# Pour installer torch consulter le site: https://pytorch.org/get-started/locally/
# exemple: pip install torch==2.0.0+cu117 torchvision==0.14.0+cu117 torchaudio==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu117 
pip install openai transformers pyaudio git+https://github.com/suno-ai/bark.git vosk numpy
```

Lastly, You also need to install the vosk model for punctuation in [alphacei website](https://alphacephei.com/vosk/models#punctuation-models). Name the folder "recasepunc" and put it in the root of the project.

## Installation

Start by cloning the repo and installing the requirements then run these commands to get started: 
```
git clone git@github.com:MysticFragilist/Convo-Buddy-GPT.git
cd Convo-Buddy-GPT
```

Run this command to setup config:
```
python setup.py
```

Make sure to provide your openai api key in the input asked:
```
OpenAI API key: sk-<your key>
```

Then, search among the list of microphones and find the index of the microphone you want to use:
```
Microphone index: <your mic index>
```

## Usage

To start the program, run:
```
python main.py
```

## Roadmap
Future improvements:
- [ ] Fine tuned bark voice to fit a specific voice and prevent coarse sound.
- [ ] Add a way to trigger response only when message is finished recording (currently capped upon a 10sec delay). 

and some more...

## License
This repository is released under the CC-BY-NC 4.0. license as found in the [LICENSE](./LICENSE) file.

## Attributions
Special thanks to multiple project to make it happened:

Thanks to suno-ai, gpt response text to voice can be generated: 
[suno-ai/bark](https://github.com/suno-ai/bark)

Thanks to vosk, voice to text can be process and sent to gpt and their recase punctuation model to polish the message:
[alphacep/vosk](https://github.com/alphacep/vosk)

Thanks to openai, gpt can be used to generate text using your own openAI key:
[openai/openai-python](https://github.com/openai/openai-python)