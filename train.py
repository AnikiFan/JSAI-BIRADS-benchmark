import torch
import torchvision
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_precision, multiclass_f1_score

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from TDSNet import TDSNet
from model4compare.GoogleNet import GoogleNet
from model4compare.AlexNet import AlexNet
from model4compare.VGG import VGG
from model4compare.NiN import NiN


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    running_precision = 0.
    running_f1 = 0.
    last_loss = 0.
    last_precision = 0.
    last_f1 = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        running_precision += multiclass_precision(outputs, labels).tolist()
        running_f1 += multiclass_f1_score(outputs, labels, average='macro').tolist()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            last_precision = running_precision / 1000  # loss per batch
            last_f1 = running_f1 / 1000  # loss per batch
            print('  batch {} loss     : {}'.format(i + 1, last_loss))
            print('  batch {} precision: {}'.format(i + 1, last_precision))
            print('  batch {} f1       : {}'.format(i + 1, last_f1))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            tb_writer.add_scalar('Precision/train', last_precision, tb_x)
            tb_writer.add_scalar('F1/train', last_f1, tb_x)
            running_loss = 0.
            running_precision = 0.
            running_f1 = 0.

    return last_loss, last_precision, last_f1


def modelSelector(model, lr, num_class):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if model == 'TDSNet':
        model = TDSNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/TDS_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'AlexNet':
        model = AlexNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/AlexNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'GoogleNet':
        model = GoogleNet(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/GoogleNet_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'VGG':
        model = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/VGG_' + str(lr) + "_" + timestamp), optimizer, timestamp
    elif model == 'NiN':
        model = NiN(num_class)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return model, SummaryWriter('runs/NiN_' + str(lr) + "_" + timestamp), optimizer, timestamp


if __name__ == '__main__':
    "----------------------------------- data ---------------------------------------------"
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256)),
         transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True, pin_memory=True,
                                                  pin_memory_device='cuda' if torch.cuda.is_available() else '')
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False, pin_memory=True,
                                                    pin_memory_device='cuda' if torch.cuda.is_available() else '')

    # Class labels
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    "----------------------------------- loss function ---------------------------------------------"
    loss_fn = torch.nn.CrossEntropyLoss()

    "----------------------------------- model ---------------------------------------------"
    model, writer, optimizer, timestamp = modelSelector('GoogleNet', 0.001, 10)

    "----------------------------------- training ---------------------------------------------"
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss, avg_precision, avg_f1 = train_one_epoch(epoch_number, writer)

        running_vloss = 0.0
        running_vprecison = 0.0
        running_vf1 = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                if torch.cuda.is_available():
                    vinputs = vinputs.to(torch.device('cuda'))
                    vlabels = vlabels.to(torch.device('cuda'))
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                vprecision = multiclass_precision(voutputs, vlabels).tolist()
                vf1 = multiclass_f1_score(voutputs, vlabels, average='macro').tolist()
                running_vloss += vloss
                running_vprecison += vprecision
                running_vf1 += vf1

        avg_vloss = running_vloss / (i + 1)
        avg_vprecision = running_vprecison / (i + 1)
        avg_vf1 = running_vf1 / (i + 1)
        print('LOSS      train {} valid {}'.format(avg_loss, avg_vloss))
        print('PRECISION train {} valid {}'.format(avg_precision, avg_vprecision))
        print('F1        train {} valid {}'.format(avg_f1, avg_vf1))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation Precision',
                           {'Training': avg_precision, 'Validation': avg_vprecision},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation F1',
                           {'Training': avg_f1, 'Validation': avg_vf1},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1
