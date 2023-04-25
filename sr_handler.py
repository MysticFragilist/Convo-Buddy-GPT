import subprocess
import json
from vosk import SetLogLevel, Model, KaldiRecognizer
import gpt_handler
import audio_handler
SetLogLevel(-1)

# must match frame rate of audio_handler.py
FRAME_RATE = 16000

model = Model(model_name="vosk-model-en-us-0.22")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

# using vosk for speech recognition transcribe the content of the queue when it is not empty
def speech_recognition(messages, recordings):
    while not messages.empty():
        frames = recordings.get()
        rec.AcceptWaveform(b''.join(frames))
        result = rec.Result()
        text = json.loads(result)['text']
        if text == "":
            print ("skip empty text")
            continue
        print("Me: " + text)
        res = gpt_handler.process_response(text)
        print("GPT:" + res)
        audio_handler.play_audio(res)

def predict(text):
    # TODO Call this from direct access throught python import code to improve realtime efficiency
    # cased = generate_predictions(config=Config(), text=text, checkpoint_path="recasepunc/checkpoint")
    cased = subprocess.check_output("python recasepunc/recasepunc.py predict recasepunc/checkpoint", shell=True, text=True, input=text)
    return cased



### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
### ================================= RecasePunc ================================= ###
### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
### ============================================================================== ###
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer, BertTokenizer

default_config = argparse.Namespace(
    seed=871253,
    lang='en',
    #flavor='flaubert/flaubert_base_uncased',
    flavor=None,
    max_length=256,
    batch_size=16,
    updates=24000,
    period=1000,
    lr=1e-5,
    dab_rate=0.1,
    device='cuda',
    debug=False
)

default_flavors = {
    'fr': 'flaubert/flaubert_base_uncased',
    'en': 'bert-base-uncased',
    'zh': 'ckiplab/bert-base-chinese',
    'tr': 'dbmdz/bert-base-turkish-uncased',
    'de': 'dbmdz/bert-base-german-uncased',
    'pt': 'neuralmind/bert-base-portuguese-cased'
}

class Config(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in default_config.__dict__.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

        assert self.lang in ['fr', 'en', 'zh', 'tr', 'pt', 'de']

        if 'lang' in kwargs and ('flavor' not in kwargs or kwargs['flavor'] is None):
            self.flavor = default_flavors[self.lang]

        #print(self.lang, self.flavor)
    


def init_random(seed):
    # make sure everything is deterministic
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# modification of XLM bpe tokenizer for keeping case information when vocab is lowercase
# forked from https://github.com/huggingface/transformers/blob/cd56f3fe7eae4a53a9880e3f5e8f91877a78271c/src/transformers/models/xlm/tokenization_xlm.py
def bpe(self, token):
    def to_lower(pair):
      #print('  ',pair)
      return (pair[0].lower(), pair[1].lower())

    from transformers.models.xlm.tokenization_xlm import get_pairs

    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    if token in self.cache:
        return self.cache[token]
    pairs = get_pairs(word)

    if not pairs:
        return token + "</w>"

    while True:
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(to_lower(pair), float("inf")))
        #print(bigram)
        if to_lower(bigram) not in self.bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
            except ValueError:
                new_word.extend(word[i:])
                break
            else:
                new_word.extend(word[i:j])
                i = j

            if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    word = " ".join(word)
    if word == "\n  </w>":
        word = "\n</w>"
    self.cache[token] = word
    return word

# modification of the wordpiece tokenizer to keep case information even if vocab is lower cased
# forked from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100, keep_case=True):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word
        self.keep_case = keep_case

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.
        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in text.strip().split():
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    # optionaly lowercase substring before checking for inclusion in vocab
                    if (self.keep_case and substr.lower() in self.vocab) or (substr in self.vocab):
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def init(config):
    init_random(config.seed)
    
    if config.lang == 'fr':
        config.tokenizer = tokenizer = AutoTokenizer.from_pretrained(config.flavor, do_lower_case=False)

        from transformers.models.xlm.tokenization_xlm import XLMTokenizer
        assert isinstance(tokenizer, XLMTokenizer)

        # monkey patch XLM tokenizer
        import types
        tokenizer.bpe = types.MethodType(bpe, tokenizer)
    else:
        # warning: needs to be BertTokenizer for monkey patching to work
        config.tokenizer = tokenizer = BertTokenizer.from_pretrained(config.flavor, do_lower_case=False) 

        # warning: monkey patch tokenizer to keep case information 
        #from recasing_tokenizer import WordpieceTokenizer
        config.tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token)

    if config.lang == 'fr':
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.bos_token_id
        config.cls_token = tokenizer.bos_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token
    else:
        config.pad_token_id = tokenizer.pad_token_id
        config.cls_token_id = tokenizer.cls_token_id
        config.cls_token = tokenizer.cls_token
        config.sep_token_id = tokenizer.sep_token_id
        config.sep_token = tokenizer.sep_token

    if not torch.cuda.is_available() and config.device == 'cuda':
        print('WARNING: reverting to cpu as cuda is not available', file=sys.stderr)
    config.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    

punctuation = {
    'O': 0,
    'COMMA': 1,
    'PERIOD': 2,
    'QUESTION': 3,
    'EXCLAMATION': 4,
}

punctuation_syms = ['', ',', '.', ' ?', ' !']

case = {
    'LOWER': 0,
    'UPPER': 1,
    'CAPITALIZE': 2,
    'OTHER': 3,
}

mapped_punctuation = {
    '.': 'PERIOD',
    '...': 'PERIOD',
    ',': 'COMMA',
    ';': 'COMMA',
    ':': 'COMMA',
    '(': 'COMMA',
    ')': 'COMMA',
    '?': 'QUESTION',
    '!': 'EXCLAMATION',
    '，': 'COMMA',
    '！': 'EXCLAMATION',
    '？': 'QUESTION',
    '；': 'COMMA',
    '：': 'COMMA',
    '（': 'COMMA',
    '(': 'COMMA',
    '）': 'COMMA',
    '［': 'COMMA',
    '］': 'COMMA',
    '【': 'COMMA',
    '】': 'COMMA',
    '└': 'COMMA',
    '└ ': 'COMMA',
    '_': 'O',
    '。': 'PERIOD',
    '、': 'COMMA', # enumeration comma
    '、': 'COMMA',
    '…': 'PERIOD',
    '—': 'COMMA',
    '「': 'COMMA',
    '」': 'COMMA',
    '．': 'PERIOD',
    '《': 'O',
    '》': 'O',
    '，': 'COMMA',
    '“': 'O',
    '”': 'O',
    '"': 'O',
    '-': 'O',
    '-': 'O',
    '〉': 'COMMA',
    '〈': 'COMMA',
    '↑': 'O',
    '〔': 'COMMA',
    '〕': 'COMMA',
}

def recase(token, label):
    if label == case['LOWER']:
        return token.lower()
    elif label == case['CAPITALIZE']:
        return token.lower().capitalize()
    elif label == case['UPPER']:
        return token.upper()
    else:
        return token

def generate_predictions(config, text, checkpoint_path):
    print(config.device)
    print(torch.cuda.is_available())
    print(config.device if torch.cuda.is_available() else 'cpu')
    loaded = torch.load(checkpoint_path, map_location=config.device if torch.cuda.is_available() else 'cpu')
    if 'config' in loaded:
        config = Config(**loaded['config'])
        init(config)

    model = Model(config.flavor, config.device)
    model.load_state_dict(loaded['model_state_dict'])

    rev_case = {b: a for a, b in case.items()}
    rev_punc = {b: a for a, b in punctuation.items()}

    complete_phrase = ''

    for line in text:
        # also drop punctuation that we may generate
        line = ''.join([c for c in line if c not in mapped_punctuation])
        if config.debug:
            print(line)
        tokens = [config.cls_token] + config.tokenizer.tokenize(line) + [config.sep_token]
        if config.debug:
            print(tokens)
        previous_label = punctuation['PERIOD']
        first_time = True
        was_word = False
        for start in range(0, len(tokens), config.max_length):
            instance = tokens[start: start + config.max_length]
            ids = config.tokenizer.convert_tokens_to_ids(instance)
            #print(len(ids), file=sys.stderr)
            if len(ids) < config.max_length:
                ids += [config.pad_token_id] * (config.max_length - len(ids))
            x = torch.tensor([ids]).long().to(config.device)
            y_scores1, y_scores2 = model(x)
            y_pred1 = torch.max(y_scores1, 2)[1]
            y_pred2 = torch.max(y_scores2, 2)[1]
            for id, token, punc_label, case_label in zip(ids, instance, y_pred1[0].tolist()[:len(instance)], y_pred2[0].tolist()[:len(instance)]):
                if config.debug:
                    print(id, token, punc_label, case_label, file=sys.stderr)
                if id == config.cls_token_id or id == config.sep_token_id:
                    continue
                if previous_label != None and previous_label > 1:
                    if case_label in [case['LOWER'], case['OTHER']]:
                        case_label = case['CAPITALIZE']
                previous_label = punc_label
                # different strategy due to sub-lexical token encoding in Flaubert
                if config.lang == 'fr':
                    if token.endswith('</w>'):
                        cased_token = recase(token[:-4], case_label)
                        if was_word:
                            complete_phrase += ' '
                        complete_phrase += cased_token + punctuation_syms[punc_label]
                        was_word = True
                    else:
                        cased_token = recase(token, case_label)
                        if was_word:
                            complete_phrase += ' '
                        complete_phrase += cased_token
                        was_word = False
                else:
                    if token.startswith('##'):
                        cased_token = recase(token[2:], case_label)
                        complete_phrase += cased_token
                    else:
                        cased_token = recase(token, case_label)
                        if not first_time:
                            complete_phrase += ' '
                        first_time = False
                        complete_phrase += cased_token + punctuation_syms[punc_label]
        if previous_label == 0:
            complete_phrase += '.'
        complete_phrase += '\n'
    return complete_phrase
