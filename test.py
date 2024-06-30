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
    with torch.no_grad():
        output = model(input)
    output = torch.sigmoid(output)
    prediction =  output.argmax(-1)
    y_pred.append(prediction.item())


acc = accuracy_score(y_true, y_pred)
print(acc)