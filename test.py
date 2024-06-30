from architecture import model
import torch
import config
from split import test_loader, validation_loader
from sklearn.metrics import accuracy_score


model.load_state_dict(torch.load("model.pth"))

y_true = []
y_pred = []
torch.manual_seed(333)

#from my a8_ex1
for input, target, _, _ in test_loader:
    model.eval()
    y_true.append(target.item())
    output = model(input)
    output = torch.sigmoid(output)
    predictions = torch.topk(output, k=1, sorted=False)[1]
    y_pred.append(predictions.item())


acc = accuracy_score(y_true, y_pred)
print(acc)