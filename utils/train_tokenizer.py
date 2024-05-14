from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

VOCAB_SIZE = 256
FILE_NAME = "harry_1_to_4"
DATA_PATH = FILE_NAME + ".txt"

tokenizer = Tokenizer(BPE(unk_token="<unk>"))

trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>"])

tokenizer.pre_tokenizer = ByteLevel()

with open(DATA_PATH, "r") as f:
    text = f.read()

texts = [text]

tokenizer.train_from_iterator(texts, trainer)

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[("<s>", 1), ("</s>", 2)]
)

tokenizer.decoder = decoders.ByteLevel()

import random
encoded = tokenizer.encode(text).ids
print(f"Tokens: {len(encoded)}")

'''
decoded_text = tokenizer.decode(encoded)

post_processed_text = decoded_text.replace('Ġ', ' ').replace('Ċ', '\n')


print(f'Post-processed text:\n{post_processed_text}')

tokenizer.save(f"{FILE_NAME}_{VOCAB_SIZE}_vocab.json")
'''