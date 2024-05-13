from utils.tokenizer import Tokenizer
from modules.mamba_llm import MambaLLM
import argparse
import yaml
import torch
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--string", type=str, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--output", type=str, default="stdout")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    model = MambaLLM(num_tokens=config["model"]["num_tokens"],
                     d_model=config["model"]["d_model"],
                     n_layers=config["model"]["n_layers"],
                     expansion=config["model"]["expansion"],
                     d_hidden=config["model"]["d_hidden"],
                     dt_min=config["model"]["dt_min"],
                     dt_max=config["model"]["dt_max"],
                     kernel_size=config["model"]["kernel_size"])
    
    # model= torch.nn.DataParallel(model)
    model.to(device)
    if args.checkpoint is not None:
        if isinstance(model, torch.nn.DataParallel):
            model.load_state_dict(torch.load(args.checkpoint))
        else:
            state_dict = torch.load(args.checkpoint)
            # Get rid of the "module." prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            model.load_state_dict(new_state_dict)

    logger.info("Number of parameters in model")
    logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))

    if os.path.exists(args.string):
        with open(args.string, "r") as f:
            string = f.read()
    else:
        string = args.string
    
    tokenizer = Tokenizer(config["tokenizer"]["file_path"])
    tokens = tokenizer.tokenize(string)
    tokens = torch.tensor(tokens, dtype=torch.int).unsqueeze(0).to(device)
    model.eval()
    output = []
    with torch.no_grad():
        logits = model(x=tokens, cache=True)
        new_token = torch.argmax(logits)
        output.append(new_token.item())
        for i in range(args.length - 1):
            logits = model(x=new_token.reshape((1,1)), one_step=True)
            new_token = torch.argmax(logits)
            output.append(new_token.item())
        logger.info("Generated text")
        logger.info(" ".join([str(i) for i in output]))
    output = tokenizer.detokenize(output)
    print(type(output))
    if args.output == "stdout":
        print(output)
    else:
        with open(args.output, "w") as f:
            f.write(output)
