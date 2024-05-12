from modules.mamba_llm import MambaLLM
from utils.data import get_text, get_dataset
from utils.tokenizer import Tokenizer
from torch.utils.data import DataLoader
import torch.utils.data
import argparse
import yaml
import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train(config: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_txt = get_text(config["data"]["url"])
    tokenizer = Tokenizer(config["tokenizer"]["file_path"])
    model = MambaLLM(num_tokens=config["model"]["num_tokens"],
                     d_model=config["model"]["d_model"],
                     n_layers=config["model"]["n_layers"],
                     expansion=config["model"]["expansion"],
                     d_hidden=config["model"]["d_hidden"],
                     dt_min=config["model"]["dt_min"],
                     dt_max=config["model"]["dt_max"],
                     kernel_size=config["model"]["kernel_size"])
    model= torch.nn.DataParallel(model)
    model.to(device)
    if config["checkpoint"] is not None:
        model.load_state_dict(torch.load(config["checkpoint"]))

    logger.info("Number of parameters in model")
    logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))

    dataset = get_dataset(data_txt, tokenizer, max_length=config["data"]["max_length"])
    tr, val = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
    tr_dataloader = DataLoader(tr, batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config["data"]["batch_size"], num_workers=config["data"]["num_workers"], shuffle=False)
    torch.autograd.set_detect_anomaly(True)

    os.makedirs(config["train"]["save_dir"], exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=config["train"]["factor"], patience=config["train"]["patience"], verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()

    logger.info("Starting training")
    for epoch in range(config["train"]["epochs"]):

        # Training loop
        model.train()
        for i, (text, label) in enumerate(tr_dataloader, start=1):
            text, label = text.to(device), label.to(device)
            model.zero_grad()
            logits = model(text)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            if i % config["train"]["log_interval"] == 0:
                logger.info(f"Train epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
                scheduler.step(loss)
            if i % config["train"]["save_interval"] == 0:
                torch.save(model.state_dict(), os.path.join(config["train"]["save_dir"], f"model_{epoch}_{i}.pt"))
        logger.info(f"Train epoch {epoch} completed, loss: {loss.item()}")
        torch.save(model.state_dict(), os.path.join(config["train"]["save_dir"], f"model_{epoch}_{i}.pt"))        
        
        # Validation loop
        model.eval()
        for i, (text, label) in enumerate(val_dataloader, start=1):
            text, label = text.to(device), label.to(device)
            with torch.no_grad():
                logits = model(text)
                loss = loss_fn(logits, label)
            if i % config["train"]["log_interval"] == 0:
                logger.info(f"Val epoch {epoch}, Iteration {i}, Loss: {loss.item()}")
        logger.info(f"Val epoch {epoch} completed, loss: {loss.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["checkpoint"] = args.checkpoint
    train(config)
