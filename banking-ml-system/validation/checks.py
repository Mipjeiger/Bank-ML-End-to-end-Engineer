from deepchecks.tabular.checks import (
    DataLeakage,
    TrainTestFeatureDrift,
    LabelImbalance
)

def get_checks():
    return [
        DataLeakage(),
        TrainTestFeatureDrift(),
        LabelImbalance()
    ]