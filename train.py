from architecture import model
import torch
from split import validation_loader,training_loader, set_seed
from tqdm import tqdm 

# from my a5_ex1
def train(model, training_loader, optimizer, show_progress, i):
    model.train()

    loop_train = tqdm(training_loader, desc=f"train epoch {i}") if show_progress else training_loader

    for input, target, _,_ in loop_train:
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        
        loss = loss_f(output, target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

def eval(model, validation_loader):
    model.eval()
    loss_eval = 0
    nr_batches_eval = 0

    with torch.no_grad():
        for input, target, _, _ in validation_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_f(output, target)

            loss_eval += loss.item()
            nr_batches_eval+=1

    average_loss_eval_i = loss_eval / nr_batches_eval
    return average_loss_eval_i

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_loss = None
loss_f = torch.nn.CrossEntropyLoss()
lr =0.00009
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.003)

# assignment description didn't say if we're to wrap the code execution blocks in an if name main, 
# but the rest of the assignments for the semester did so, so I'm doing it here as well, 
# I hope it won't be a problem 
if __name__ == '__main__':
    set_seed(333)
    # from hoai u6
    for epoch in range(30):
        train(model, training_loader, optimizer, True, epoch) 
        val_loss = eval(model, validation_loader)

        print("-" * 50)
        print(f"| end of epoch {epoch:3d}", f"| valid loss {val_loss:5.2f}")
        print("-" * 50)
        
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), "model.pth")
            best_val_loss = val_loss
        else:
            lr /= 10.0
            for g in optimizer.param_groups:
                g["lr"] = lr