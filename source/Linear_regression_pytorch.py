import torch
import pandas as pd
from torch.autograd import Variable

x = pd.DataFrame([[1.0,2.0],[3.0,4.0],[5.0,6.0]])
y = pd.DataFrame([1,2,3])
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
x_data = torch.Tensor(x.values)
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
y_data = torch.Tensor(y.values)

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # One in and one out
 
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

our_model = LinearRegressionModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)

for epoch in range(10):

    # Forward pass: Compute predicted y by passing
    # x to the model
    pred_y = our_model(x_data)
    # print(f'pred dim: {pred_y.shape}, y_data dim: {y_data.shape}')
    # Compute and print loss
    loss = criterion(pred_y, y_data)
 
    # Zero gradients, perform a backward pass,
    # and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

new_var = Variable(torch.Tensor([[4.0,5.0]]))
new_var = Variable(torch.Tensor([[4.0,5.0]]))
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())