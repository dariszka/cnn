from architecture import model
import torch
import config
from split import test_loader,validation_loader,training_loader
from tqdm import tqdm 

# from my a5_ex1
def train(model, training_loader, optimizer, show_progress, i):
    model.train()
    minibatch_loss_train = 0
    nr_batches_train = 0

    loop_train = tqdm(training_loader, desc=f"train epoch {i}") if show_progress else training_loader

    for input, target, _,_ in loop_train:
        output = model(input)
        loss = loss_f(output, target)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        nr_batches_train+=1

        # print('Minibatch nr: ',nr_batches_train, 'Loss: ', loss.item())

def eval(model, validation_loader):
    model.eval()
    loss_eval = 0
    nr_batches_eval = 0

    with torch.no_grad():
        for input, target, _, _ in validation_loader:
            output = model(input)
            loss = loss_f(output, target)

            loss_eval += loss.item()
            nr_batches_eval+=1

    average_loss_eval_i = loss_eval / nr_batches_eval
    return average_loss_eval_i


num_epochs = config.num_epochs
best_val_loss = None
loss_f = torch.nn.CrossEntropyLoss()
train_losses, val_losses = [],[]
lr =0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
torch.manual_seed(333)


# from hoai u6
for epoch in range(10):
    train(model, training_loader, optimizer, True, epoch) 
    val_loss = eval(model, validation_loader)

    print("-" * 100)
    print(f"| end of epoch {epoch:3d}", f"| valid loss {val_loss:5.2f}")
    print("-" * 100)
    
    if not best_val_loss or val_loss < best_val_loss:
        torch.save(model.state_dict(), "model.pth")
        best_val_loss = val_loss
    else:
        lr /= 4.0
        for g in optimizer.param_groups:
            g["lr"] = lr