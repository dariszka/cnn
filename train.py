from architecture import model
import torch
import config
from split import test_loader,validation_loader,training_loader

# from my a5_ex1
def train(model, training_loader, optimizer):
    model.train()
    minibatch_loss_train = 0
    nr_batches_train = 0

    for input, target, _,_ in training_loader:
        output = model(input)
        loss = loss_f(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        nr_batches_train+=1

        print('Minibatch nr: ',nr_batches_train, 'Loss: ', loss.item())

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


# from hoai u6
num_epochs = config.num_epochs
best_val_loss = None
loss_f = torch.nn.CrossEntropyLoss()
train_losses, val_losses = [],[]
lr =0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
torch.manual_seed(333)


for epoch in range(3):
    train(model, training_loader, optimizer) 
    val_loss = eval(model, validation_loader)

    print("-" * 100)
    print(f"| end of epoch {epoch:3d}", f"| valid loss {val_loss:5.2f}")
    print("-" * 100)
    
    if not best_val_loss or val_loss < best_val_loss:
        torch.save(model.state_dict(), "model.pth")
        best_val_loss = val_loss
    else:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        lr /= 4.0
        for g in optimizer.param_groups:
            g["lr"] = lr