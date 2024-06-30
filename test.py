from architecture import model
import torch
from split import test_loader
from sklearn.metrics import accuracy_score

model.load_state_dict(torch.load("model.pth"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

y_true = []
y_pred = []

#from my a8_ex1
def test():
    for input, target, _, _ in test_loader:
        input = input.to(device)
        target = target.to(device)
        y_true.append(target.item())

        model.eval()
        with torch.no_grad():
            output = model(input)
        output = torch.sigmoid(output)

        prediction =  output.argmax(-1)
        y_pred.append(prediction.item())

if __name__ == '__main__':
    torch.manual_seed(333)
    test()
    acc = accuracy_score(y_true, y_pred)

    print(acc)