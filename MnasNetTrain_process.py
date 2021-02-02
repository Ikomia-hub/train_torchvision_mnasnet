from ikomia import core, dataprocess
from ikomia.dnn import dnntrain
import copy
# Your imports below
import MnasNet


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class MnasNetTrainParam(dataprocess.CDnnTrainProcessParam):

    def __init__(self):
        dataprocess.CDnnTrainProcessParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'mnasnet'
        self.batch_size = 8
        self.classes = 2
        self.epochs = 15
        self.num_workers = 0
        self.input_size = 224
        self.use_pretrained = True
        self.feature_extract = True
        self.export_pth = True
        self.export_onnx = False

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        super().setParamMap(paramMap)
        self.num_workers = int(paramMap["num_workers"])
        self.input_size = int(paramMap["input_size"])
        self.use_pretrained = bool(paramMap["use_pretrained"])
        self.feature_extract = bool(paramMap["feature_extract"])
        self.export_pth = bool(paramMap["export_pth"])
        self.export_onnx = bool(paramMap["export_onnx"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = super().getParamMap()
        param_map["num_workers"] = str(self.num_workers)
        param_map["input_size"] = str(self.input_size)
        param_map["use_pretrained"] = str(self.use_pretrained)
        param_map["feature_extract"] = str(self.feature_extract)
        param_map["export_pth"] = str(self.export_pth)
        param_map["export_onnx"] = str(self.export_onnx)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class MnasNetTrainProcess(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        # Add input/output of the process here
        self.addInput(dataprocess.CPathIO(core.IODataType.FOLDER_PATH))

        # Create parameters class
        if param is None:
            self.setParam(MnasNetTrainParam())
        else:
            self.setParam(copy.deepcopy(param))

        self.trainer = MnasNet.Mnasnet(self.getParam())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.epochs
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

    def on_epoch_end(self, metrics):
        # Step progress bar:
        self.emitStepProgress()
        # Log metrics
        self.log_metrics(metrics)

    def stop(self):
        super().stop()
        self.trainer.stop()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class MnasNetTrainProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "MnasNetTrain"
        self.info.shortDescription = "Training process for MnasNet convolutional network."
        self.info.description = "Training process for MnasNet convolutional network. It requires a specific dataset " \
                                "structure based on folder names. It follows the PyTorch torchvision convention. " \
                                "The process enables to train ResNet network from scratch or for transfer learning. " \
                                "One could train the full network from pre-trained weights or keep extracted features " \
                                "and re-train only the classification layer."
        self.info.authors = "Ikomia"
        self.info.version = "1.0.0"
        self.info.year = 2020
        self.info.license = "MIT License"
        self.info.repo = "https://github.com/Ikomia-dev"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Train"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.keywords = "MnasNet,classification,train,mobile,edge"

    def create(self, param=None):
        # Create process object
        return MnasNetTrainProcess(self.info.name, param)
