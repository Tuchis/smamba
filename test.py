from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders




# Set decoder
tokenizer = Tokenizer.from_file("utils/fancy_tokenizer.json")

# Tokenize the text

# Print tokens and total count
#print(f'Tokens: {encoded.tokens}')
#print(f'Total tokens: {len(encoded.tokens)}')

import random 


with open("text.txt", "r") as f:
    text = f.read()
encoded = tokenizer.encode(text[:100])
decoded = tokenizer.decode(encoded.ids)
print("tokens type: {}".format(type(decoded)))
decoded = decoded.split(' ')
decoded = [ token.replace('Ġ', ' ').replace('Ċ', '\n') for token in decoded]
decoded = ''.join(decoded)
print(f"Decoded: {decoded}")

print(f"Space token: {tokenizer.token_to_id('Ġ')}")
