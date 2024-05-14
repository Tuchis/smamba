import tokenizers
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.trainers

class Tokenizer:
    def __init__(self, file_path: str = None):
        if file_path:
            self.tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE()).from_file(file_path)
        else:
            self.tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
            self.tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    def tokenize(self, text):
        return self.tokenizer.encode(text).ids

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def train(self, data: str, file_path: str = None, vocab_size: int = 100):
        trainer = tokenizers.trainers.BpeTrainer(vocab_size=vocab_size)
        self.tokenizer.train_from_iterator(data.splitlines(), trainer=trainer)
        if file_path:
            self.tokenizer.save(file_path)

    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())
    

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers import decoders

class FancyTokenizer:
    def __init__(self, file_path: str = None):
        if file_path:
            self.tokenizer = Tokenizer.from_file(file_path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = ByteLevel()
            self.tokenizer.decoder = decoders.ByteLevel()
            self.tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                pair="<s> $A </s> $B:1 </s>:1",
                special_tokens=[("<s>", 1), ("</s>", 2)]
            )

    def tokenize(self, text):
        return self.tokenizer.encode(text).ids

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens)
        print("tokens type: {}".format(type(tokens)))
        decoded_text = self.tokenizer.decode(tokens)
        return "".join(decoded_text)#.replace('Ċ', '\n').replace('Ġ', ' ')

    def train(self, data: str, file_path: str = None, vocab_size: int = 100):
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>"])
        self.tokenizer.train_from_iterator(data.splitlines(), trainer=trainer)
        if file_path:
            self.tokenizer.save(file_path)

    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

def postprocess(text: str):
    decoded = text.split(' ')
    decoded = [ token.replace('Ġ', ' ').replace('Ċ', '\n') for token in decoded]
    return ''.join(decoded)

if __name__ == "__main__":
    tokenizer = Tokenizer()
    from data import get_text
    data = get_text("https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt")
    tokenizer.train(data, "tokenizer.json")
