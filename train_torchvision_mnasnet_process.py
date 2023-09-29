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

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["batch_size"] = int(param_map["batch_size"])
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
        self.remove_input(0)
        self.add_input(dataprocess.CPathIO(core.IODataType.FOLDER_PATH))

        # Create parameters class
        if param is None:
            self.set_param_object(TrainMnasnetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.trainer = mnasnet.Mnasnet(self.get_param_object())
        self.enable_tensorboard(False)

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.get_param_object()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get dataset path from input
        path_input = self.get_input(0)

        print("Starting training job...")
        self.trainer.launch(path_input.get_path(), self.on_epoch_end)

        print("Training job finished.")

        # Call end_task_run to finalize process
        self.end_task_run()

    def on_epoch_end(self, metrics, epoch):
        # Step progress bar:
        self.emit_step_progress()
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
        self.info.short_description = "Training process for MnasNet convolutional network."
        self.info.authors = "Ikomia"
        self.info.version = "1.3.1"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.repository = "https://github.com/Ikomia-hub/train_torchvision_mnasnet"
        self.info.original_repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.keywords = "MnasNet,classification,train,mobile,edge"
        self.info.algo_type = core.AlgoType.TRAIN
        self.info.algo_tasks = "CLASSIFICATION"

    def create(self, param=None):
        # Create process object
        return TrainMnasnet(self.info.name, param)
