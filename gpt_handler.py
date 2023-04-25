import openai
import os
import config_handler

openai.api_key = config_handler.getConfig('openaikey')

currentConvo = ""

def process_response(text):
  global currentConvo
  currentConvo = currentConvo + "\nMe: " + text
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt="Respond to the following conversation using only text you would say to the person talking and nothing else. Don't say 'PERSON:', 'GPT:' or any kind of meta information on the text. Here is the script:\n" + currentConvo,
    temperature=0.7,
    max_tokens=709,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  currentConvo = currentConvo + "\nGpt: " + response["choices"][0]["text"]
  return response["choices"][0]["text"].replace("\n", " ").replace("Gpt: ", "").replace("GPT: ", "").replace("Me: ", "").replace("PERSON: ", "")