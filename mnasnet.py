import os
import copy
import time
import torch
from datetime import datetime
from ikomia.dnn.torch import utils, models
from torchvision import datasets, transforms


class Mnasnet:
    def __init__(self, parameters):
        self.stop_train = False
        self.parameters = parameters
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_data(self, dataset_dir):
        print("Initializing Datasets and Dataloaders...")

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.parameters.cfg["input_size"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(self.parameters.cfg["input_size"]),
                transforms.CenterCrop(self.parameters.cfg["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create training and validation datasets
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) for x in ['train', 'val']
        }
        # Create training and validation dataloaders
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.parameters.cfg["batch_size"],
                                           shuffle=True,
                                           num_workers=self.parameters.cfg["num_workers"]) for x in ['train', 'val']
        }
        return dataloaders_dict

    def init_optimizer(self, model):
        # Send the model to GPU if available
        model_ft = model.to(self.device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.parameters.cfg["feature_extract"]:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.SGD(params_to_update, lr=self.parameters.cfg["learning_rate"],
                                       momentum=self.parameters.cfg["momentum"])
        return optimizer_ft

    def train_model(self, model, data_loaders, criterion, optimizer, on_epoch_end):
        since = time.time()

        metrics = {}
        val_acc_history = []
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epoch_loss = 0.0
        epoch_acc = 0.0

        for epoch in range(self.parameters.cfg["epochs"]):
            print('Epoch {}/{}'.format(epoch + 1, self.parameters.cfg["epochs"]))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)
                    metrics["Loss"] = epoch_loss
                    metrics["Accuracy"] = epoch_acc.item()
                    metrics["Best accuracy"] = best_acc.item()
                    on_epoch_end(metrics)

            print()
            if self.stop_train:
                break

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best accuracy: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def launch(self, dataset_dir, on_epoch_end):
        self.stop_train = False

        # Load dataset
        data_loaders = self.load_data(dataset_dir)

        # Setup the loss function
        loss = torch.nn.CrossEntropyLoss()

        # Initialize the model for this run
        model = models.mnasnet(train_mode=True, use_pretrained=self.parameters.cfg["use_pretrained"],
                               feature_extract=self.parameters.cfg["feature_extract"],
                               classes=self.parameters.cfg["classes"])

        optimizer = self.init_optimizer(model)

        # Train and evaluate
        model_ft, hist = self.train_model(model, data_loaders, loss, optimizer, on_epoch_end)

        # Save model
        if not os.path.isdir(self.parameters.cfg["output_folder"]):
            os.mkdir(self.parameters.cfg["output_folder"])

        if not self.parameters.cfg["output_folder"].endswith('/'):
            self.parameters.cfg["output_folder"] += '/'

        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        model_folder = self.parameters.cfg["output_folder"] + str_datetime + "/"

        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        # .pth
        if self.parameters.cfg["export_pth"]:
            model_path = model_folder + self.parameters.cfg["model_name"] + ".pth"
            utils.save_pth(model_ft, model_path)

        # .onnx
        if self.parameters.cfg["export_onnx"]:
            model_path = model_folder + self.parameters.cfg["model_name"] + ".onnx"
            input_shape = [1, 3, self.parameters.cfg["input_size"], self.parameters.cfg["input_size"]]
            utils.save_onnx(model, input_shape, self.device, model_path)

    def stop(self):
        self.stop_train = True
