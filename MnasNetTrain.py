from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class MnasNetTrain(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from MnasNetTrain.MnasNetTrain_process import MnasNetTrainProcessFactory
        # Instantiate process object
        return MnasNetTrainProcessFactory()

    def getWidgetFactory(self):
        from MnasNetTrain.MnasNetTrain_widget import MnasNetTrainWidgetFactory
        # Instantiate associated widget object
        return MnasNetTrainWidgetFactory()
