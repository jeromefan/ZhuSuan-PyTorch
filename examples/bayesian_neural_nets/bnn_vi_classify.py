import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import zhusuan.bn as bn


class Bayes_LeNet5(bn.BModule):
    '''
    Create a Bayes version of LeNet-5
    '''

    def __init__(self):
        super().__init__()
        self.conv_1 = bn.BConv2d(3, 6, (5, 5))
        self.conv_2 = bn.BConv2d(6, 16, (5, 5))
        self.fc_1 = bn.BLinear(400, 120)
        self.fc_2 = bn.BLinear(120, 84)
        self.fc_3 = bn.BLinear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


def save_model(model_name, model):
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if isinstance(model_name, float):
        save_name = 'checkpoint/{:.2f}.pt'.format(model_name)
    else:
        save_name = 'checkpoint/' + model_name + '.pt'
    torch.save(model.state_dict(), save_name)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_samples = 20
    epoch_size = 20
    batch_size = 128
    learning_rate = 0.005
    data_path = 'data/cifar10'
    complexity_cost_weight_type = 'Blundell'

    train_datasets = datasets.CIFAR10(data_path, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                      ]))
    test_datasets = datasets.CIFAR10(data_path, train=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                     ]))

    train_loader = tud.DataLoader(
        train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = tud.DataLoader(
        test_datasets, batch_size=batch_size)

    model = Bayes_LeNet5()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    highest_acc = 0.6
    for epoch in range(epoch_size):
        pbar_training = tqdm(train_loader)
        model.train()
        for i, (data, target) in enumerate(pbar_training):
            pbar_training.set_description(f'Epoch={epoch+1} training')
            data, target = data.to(device), target.to(device)

            if complexity_cost_weight_type == 'Graves':
                loss = model.elbo_estimator(
                    data, target, n_samples, nn.CrossEntropyLoss(), len(train_datasets), batch_size)
            elif complexity_cost_weight_type == 'Blundell':
                loss = model.elbo_estimator(
                    data, target, n_samples, nn.CrossEntropyLoss(), len(train_datasets), batch_size, i + 1, 'Blundell')
            else:
                raise NotImplementedError()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        acc = 0.
        pbar_testing = tqdm(test_loader)
        for i, (data, target) in enumerate(pbar_testing):
            pbar_testing.set_description(f'Epoch={epoch+1} testing')
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            acc += torch.sum(preds.eq(target)).item()

        acc = acc / len(test_loader.dataset)
        print(f'acc = {100 * acc}%')
        if acc > highest_acc:
            highest_acc = acc
            save_model(highest_acc, model)
    save_model('last_checkpoint', model)

    print('Load the saved checkpoint to check')
    net = Bayes_LeNet5()
    model.load_state_dict(torch.load('checkpoint/last_checkpoint.pt'))
    model.eval()
    acc = 0.
    pbar_testing = tqdm(test_loader)
    for i, (data, target) in enumerate(pbar_testing):
        pbar_testing.set_description(f'Epoch={epoch+1} testing')
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        acc += torch.sum(preds.eq(target)).item()

    acc = acc / len(test_loader.dataset)
    print(f'acc = {100 * acc}%')
