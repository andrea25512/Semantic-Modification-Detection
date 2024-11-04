import torch.nn as nn

class TwoRegressor(nn.Module):
    def __init__(self, hidden_size):
        super(TwoRegressor, self).__init__()
        self.fc1 = nn.Linear(1024, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, 1)           

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ThreeRegressor(nn.Module):
    def __init__(self, hidden_size):
        super(ThreeRegressor, self).__init__()
        self.fc1 = nn.Linear(1024, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, 1)           

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x