import wandb


for x in range(2):
    run = wandb.init(project="my-test-project", reinit=True, save_code=True)
    for y in range(100):
        wandb.log({"metrics": x + y})
    run.finish()
