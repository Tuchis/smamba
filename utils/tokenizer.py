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
    
    def train(self, data: str, file_path: str = None, vocab_size: int = 10000):
        trainer = tokenizers.trainers.BpeTrainer(vocab_size=vocab_size)
        self.tokenizer.train_from_iterator(data.splitlines(), trainer=trainer)
        if file_path:
            self.tokenizer.save(file_path)

    def get_vocab_size(self):
        return len(self.tokenizer.get_vocab())

if __name__ == "__main__":
    tokenizer = Tokenizer()
    from data import get_text
    data = get_text("https://raw.githubusercontent.com/karpathy/ng-video-lecture/master/input.txt")
    tokenizer.train(data, "tokenizer.json")
