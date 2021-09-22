from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from train_torchvision_mnasnet.train_torchvision_mnasnet_process import TrainMnasnetFactory
        # Instantiate process object
        return TrainMnasnetFactory()

    def getWidgetFactory(self):
        from train_torchvision_mnasnet.train_torchvision_mnasnet_widget import TrainMnasnetWidgetFactory
        # Instantiate associated widget object
        return TrainMnasnetWidgetFactory()
