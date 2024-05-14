# S(m)AMBA

## Authors
- [Vladyslav Humennyy](https://github.com/Tuchis)
- [Volodymyr Kuzma](https://github.com/Kuzmarg)
- [Roman Naumenko](https://github.com/Raspy-Py)



## Description

Custom implementation of [Mamba](https://github.com/state-spaces/mamba) state space model pretrained for NLP tasks..

S(m)AMBA reflects our vision of how S4 and Mamba should look like.
We trained this model on four different datasets:
1. **Shakespeare** - collection of Shakespeare's writings.
2. **Harry Tinny** - "Boy Who Lived" chapter of the first book in the famous series.
3. **Harry 1** - entire first book.
4. **Harry Full** - first 4 books in the series.


## Usage

### Training

```bash
python train.py --config configs/<model>_config.yaml
```

For `config` file you can chose one of already available configs (in the `configs` directory), or create or own.

### Inference

For inference you can also choose from predifined configs with pretrained models.
```bash
python inference.py \
    --config configs/<model>_config.yaml \
    --checkpoint weights/<model>.pth \
    --string "Example prompt." \
    --length 256
```

## License

Unlicensed