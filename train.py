from architecture import model
import torch
import config

def train():
    pass

def eval():
    pass



# taken from hoai u6
best_val_loss = None
for epoch in range(range(config['num_epochs'])):
    train(model) 
    val_loss = eval(model)

    print("-" * 100)
    print(f"| end of epoch {epoch:3d}", f"| valid loss {val_loss:5.2f}")
    print("-" * 100)
    
    if not best_val_loss or val_loss < best_val_loss:
        with open(config['save_path'], "wb") as f:
            torch.save(model, f)
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0
        for g in optimizer.param_groups:
            g["lr"] = lr