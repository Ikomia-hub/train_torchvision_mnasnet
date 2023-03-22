from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_torchvision_mnasnet.train_torchvision_mnasnet_process import TrainMnasnetParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class TrainMnasnetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainMnasnetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.spin_workers = pyqtutils.append_spin(self.grid_layout, label="Data loader workers",
                                                  value=self.parameters.cfg["num_workers"],
                                                  min=0, max=8, step=2)

        self.spin_batch = pyqtutils.append_spin(self.grid_layout, label="Batch size",
                                                value=self.parameters.cfg["batch_size"],
                                                min=1, max=1024, step=1)

        self.spin_epoch = pyqtutils.append_spin(self.grid_layout, label="Epochs",
                                                value=self.parameters.cfg["epochs"], min=1)

        self.spin_size = pyqtutils.append_spin(self.grid_layout, label="Input size",
                                               value=self.parameters.cfg["input_size"])

        self.check_pretrained = pyqtutils.append_check(self.grid_layout, label="Pre-trained model",
                                                       checked=self.parameters.cfg["use_pretrained"])

        self.check_features = pyqtutils.append_check(self.grid_layout, label="Feature Extract mode",
                                                     checked=self.parameters.cfg["feature_extract"])

        label_model_format = QLabel("Model format")
        row = self.grid_layout.rowCount()
        self.grid_layout.addWidget(label_model_format, row, 0)
        self.check_pth = QCheckBox("pth")
        self.check_pth.setChecked(self.parameters.cfg["export_pth"])
        self.grid_layout.addWidget(self.check_pth, row, 1)
        self.check_onnx = QCheckBox("onnx")
        self.check_onnx.setChecked(self.parameters.cfg["export_onnx"])
        self.grid_layout.addWidget(self.check_onnx, row + 1, 1)

        self.browse_folder = pyqtutils.append_browse_file(self.grid_layout, label="Output folder",
                                                          path=self.parameters.cfg["output_folder"],
                                                          tooltip="Select output folder",
                                                          mode=QFileDialog.Directory)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.cfg["num_workers"] = self.spin_workers.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["epochs"] = self.spin_epoch.value()
        self.parameters.cfg["input_size"] = self.spin_size.value()
        self.parameters.cfg["use_pretrained"] = self.check_pretrained.isChecked()
        self.parameters.cfg["feature_extract"] = self.check_features.isChecked()
        self.parameters.cfg["export_pth"] = self.check_pth.isChecked()
        self.parameters.cfg["export_onnx"] = self.check_onnx.isChecked()
        self.parameters.cfg["output_folder"] = self.browse_folder.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class TrainMnasnetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_torchvision_mnasnet"

    def create(self, param):
        # Create widget object
        return TrainMnasnetWidget(param, None)
