import torch
import torch.nn as nn
import torch.optim as optim


class Mlptorch(nn.Module):
    def __init__(self):
        super(Mlptorch, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


net = Mlptorch()


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.8)

inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
outputs = torch.tensor([[0.0], [1.0], [1.0], [0.0]])


epochs = 10000

for epoch in range(epochs):
    optimizer.zero_grad()
    output = net(inputs)
    loss = criterion(output, outputs)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item()}")


with torch.no_grad():
    test_inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    test_outputs = net(test_inputs)

    print("\nTest Results:")
    print("Input    Expected    Predicted")
    for inp, exp, pred in zip(test_inputs, outputs, test_outputs):
        print(f"{inp.numpy()}    {exp.item():.4f}       {pred.item():.4f}")
