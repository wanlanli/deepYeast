import pandas as pd


from analyser.mating.mating_movie import Mating
from analyser.flourescent.fluorescent_classification import FluorescentClassification


def step_1_init_cell_mating_class(merged_data, cells, n_components=3):
    """初始化mating ideos
    """
    cells = Mating(merged_data, cells)
    # prediction flourencsent type
    data = cells.instance_fluorescent_intensity()
    fc = FluorescentClassification(data)
    _, _ = fc.predition_data_type(n_components=n_components)
    pred = fc.data.groupby('label')['channel_prediction'].agg(pd.Series.mode)
    cells.set_type(pred)
    cells.init_propoerties()
    return cells

