import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/

# load the dataset, split into input (X) and output (y) variables
# the 2102 y-values of the graph will be our input,
# and the 1 label at the end will be our output.
# TODO: ensure that our dataset has an appropriate amount of points (does our current set have too many?)
dataset = np.loadtxt('CSV_TrainingDSTable.csv', delimiter=',')
X = dataset[:,0:2102]
y = dataset[:,2102]


# convert variables to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# define the model
# TODO: configure this model to suit our dataset
model = nn.Sequential(
    nn.Linear(2102, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TODO: to edit the # of epochs and batch size
n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i + batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")