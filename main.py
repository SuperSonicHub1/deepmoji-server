# curl http://127.0.0.1:8080 --header "Content-Type: application/json" --data-raw '["I love you, John"]'

import json
import csv
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from flask import Flask, jsonify, request
import numpy as np
from typing import List, Dict

maxlen = 30
batch_size = 32

with open('emoji_unicode.csv') as f:
	emoji_map = {
		i: chr(int(c, base=16))
		for i, (_, c)
		in enumerate(csv.reader(f))
	}

print(f"Initializing tokenizer with vocabulary from {VOCAB_PATH}")
with open(VOCAB_PATH, 'r') as f:
	vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

def get_sentiment(sentences: List[str]) -> List[Dict[str, float]]:
	tokenized, _, _ = st.tokenize_sentences(sentences)
	sentence_sentimentes = model.predict(tokenized)

	return [
			{
			emoji_map[i]: float(probability)
			for i, probability
			in enumerate(sentiment)
		}
		for sentiment
		in sentence_sentimentes
	]

app = Flask(__name__)

@app.get("/")
def index_get():
	return "Send a POST request to this URL with a JSON array of sentences."

@app.post("/")
def index_post():
	sentences = request.json
	return jsonify(get_sentiment(sentences))

app.run('0.0.0.0', 8080)
