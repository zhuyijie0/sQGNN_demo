import torch.utils.data as Data
import numpy as np
import torch
import pennylane as qml
from torch import nn
from torch_geometric.datasets import TUDataset
from torch import optim
from torch.utils.data import SubsetRandomSampler
import qiskit.providers.aer.noise as noise

batch_size = 256
total_epoch = 200
lr = 0.1
classes_num = 2
testing_split = .1
shuffle_dataset = True
random_seed = 9

np_data = np.loadtxt('./data/PTC_FM.csv',delimiter=',')
data = torch.Tensor(np_data)

dataset = TUDataset(root='dataset/PTC_FM', name='PTC_FM')
Y = np.array([])
for i in range(len(dataset)):
    Y = np.append(Y,np.array(dataset[i].y)[0])


all_dataset = Data.TensorDataset(data,torch.LongTensor(Y))

dataset_size = len(all_dataset)
indices = list(range(dataset_size))
split = int(np.floor(testing_split * dataset_size))

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, test_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = Data.DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                               pin_memory=True, sampler=train_sampler)
test_loader = Data.DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                              pin_memory=True, sampler=test_sampler)

dev = qml.device("default.qubit", wires=1)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

@qml.qnode(dev)
def circuit(inputs, weights):
    qml.Rot(weights[0] + inputs[0] * weights[1], weights[2] + inputs[1] * weights[3],
            weights[4] + inputs[2] * weights[5], wires=0)
    qml.Rot(weights[6] + inputs[3] * weights[7], weights[8] + inputs[4] * weights[9],
            weights[10] + inputs[5] * weights[11], wires=0)
    qml.Rot(weights[12] + inputs[6] * weights[13], weights[14] + inputs[7] * weights[15],
            weights[16] + inputs[8] * weights[17], wires=0)
    qml.Rot(weights[18] + inputs[9] * weights[19], weights[20] + inputs[10] * weights[21],
            weights[22] + inputs[11] * weights[23], wires=0)

    return [qml.expval(qml.Hermitian([[1,0],[0,0]], wires=[0]))]

class QGN(nn.Module):
    def __init__(self):
        super().__init__()
        phi = 24

        weight_shapes = {"weights": phi}
        self.qlayer_1 = qml.qnn.TorchLayer(circuit, weight_shapes)
        #self.post_net = nn.Linear(1, 2)

    def forward(self, input_features):
        out = self.qlayer_1(input_features)
        out = torch.FloatTensor(out)
        out = out.to(device)

        return out

class FLoss(torch.nn.Module):
    def __init__(self):
        super(FLoss, self).__init__()

    def forward(self, output, target):
        x_t = torch.randn([len(output),1])
        for i in range(len(output)):
            if target[i] == 0:
                x_t[i][0] = output[i][0]
            else:
                x_t[i][0] = output[i][1]
        f_loss = (1 - x_t)**2

        return torch.mean(f_loss)

model = QGN()
model = model.to(device)
model.train()
criterion = FLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
total_step = len(train_loader)

for epoch in range(total_epoch):

    train_loss = 0
    running_loss = 0
    train_acc = 0
    num_correct = 0
    num_total = 0

    for i, loader_data in enumerate(train_loader):
        train_data, labels = loader_data
        train_data = train_data.to(device)
        labels = labels.to(device)

        outputs = model(train_data)
        outputs_2 = torch.cat([outputs, 1 - outputs], 1)
        loss = criterion(outputs_2, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        pred = outputs_2.argmax(dim=1)

        num_total += labels.size(0)
        num_correct += pred.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        accu = 100. * num_correct / num_total

        if (i + 1) % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                  .format(epoch + 1, total_epoch, i + 1, total_step, train_loss, accu))


torch.save(model.state_dict(), 'demo_PTC_FM.pkl')

test_model = QGN()
test_model.load_state_dict(torch.load('demo_train_PTC_FM.pkl'))
test_model = test_model.to(device)
test_model.eval()

test_criterion = FLoss()

print('Start testing')
print('------------------------------------------------------------------------------')

for epoch in range(1):
    with torch.no_grad():

        running_test_loss = 0
        test_total = 0
        test_correct = 0

        for i, test_data in enumerate(test_loader):
            test_input_data, test_labels = test_data

            test_input_data = test_input_data.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_input_data)
            test_outputs_2 = torch.cat([test_outputs, 1 - test_outputs], 1)

            test_loss = test_criterion(test_outputs_2, test_labels)
            test_per = test_loss.item()
            running_test_loss += test_loss.item()

            test_predicted = test_outputs_2.argmax(dim=1)

            test_total += test_labels.size(0)
            test_correct += test_predicted.eq(test_labels).sum().item()

            test_loss = running_test_loss / len(test_loader)
            test_accu = 100. * test_correct / test_total

        print('Loss: {:.4f}, Acc: {:.4f}, Loss_item: {:.4f}'.format(test_loss, test_accu, test_per))
