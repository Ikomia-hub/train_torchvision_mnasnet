from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
import os
import copy
from train_torchvision_mnasnet import mnasnet


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class TrainMnasnetParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["model_name"] = 'mnasnet'
        self.cfg["batch_size"] = 8
        self.cfg["epochs"] = 15
        self.cfg["learning_rate"] = 0.001
        self.cfg["momentum"] = 0.9
        self.cfg["weight_decay"] = 1e-4
        self.cfg["num_workers"] = 0
        self.cfg["input_size"] = 224
        self.cfg["use_pretrained"] = True
        self.cfg["feature_extract"] = True
        self.cfg["export_pth"] = True
        self.cfg["export_onnx"] = False
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/models/"

    def setParamMap(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["classes"] = int(param_map["classes"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["learning_rate"] = float(param_map["learning_rate"])
        self.cfg["momentum"] = float(param_map["momentum"])
        self.cfg["weight_decay"] = float(param_map["weight_decay"])
        self.cfg["num_workers"] = int(param_map["num_workers"])
        self.cfg["input_size"] = int(param_map["input_size"])
        self.cfg["use_pretrained"] = bool(param_map["use_pretrained"])
        self.cfg["feature_extract"] = bool(param_map["feature_extract"])
        self.cfg["export_pth"] = bool(param_map["export_pth"])
        self.cfg["export_onnx"] = bool(param_map["export_onnx"])
        self.cfg["output_folder"] = param_map["output_folder"]


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class TrainMnasnet(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        self.removeInput(0)
        self.addInput(dataprocess.CPathIO(core.IODataType.FOLDER_PATH))

        # Create parameters class
        if param is None:
            self.setParam(TrainMnasnetParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.trainer = mnasnet.Mnasnet(self.getParam())
        self.enableTensorboard(False)

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get dataset path from input
        path_input = self.getInput(0)

        print("Starting training job...")
        self.trainer.launch(path_input.getPath(), self.on_epoch_end)

        print("Training job finished.")

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def on_epoch_end(self, metrics, epoch):
        # Step progress bar:
        self.emitStepProgress()
        # Log metrics
        self.log_metrics(metrics, epoch)

    def stop(self):
        super().stop()
        self.trainer.stop()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class TrainMnasnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_torchvision_mnasnet"
        self.info.shortDescription = "Training process for MnasNet convolutional network."
        self.info.description = "Training process for MnasNet convolutional network. It requires a specific dataset " \
                                "structure based on folder names. It follows the PyTorch torchvision convention. " \
                                "The process enables to train ResNet network from scratch or for transfer learning. " \
                                "One could train the full network from pre-trained weights or keep extracted features " \
                                "and re-train only the classification layer."
        self.info.authors = "Ikomia"
        self.info.version = "1.3.0"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.repo = "https://github.com/Ikomia-dev"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.keywords = "MnasNet,classification,train,mobile,edge"

    def create(self, param=None):
        # Create process object
        return TrainMnasnet(self.info.name, param)
