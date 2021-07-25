import os
import math
import random
import wandb


def generate_random_number(step: int) -> float:
    return math.log(0.1 + random.random() + step * 0.01)


config = dict(hyperparameter=4,)
run = wandb.init(project="my-test-project", config=config, save_code=True)

with run:
    for step in range(100):
        wandb.log(
            {
                "acc": generate_random_number(step),
                "val_acc": generate_random_number(step),
                "loss": wandb.config.hyperparameter - generate_random_number(step),
                "val_loss": wandb.config.hyperparameter - generate_random_number(step),
            }
        )
