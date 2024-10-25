import pandas as pd
from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import visualisation as vis

hrtf1,hrtf2 = ld.load_example_sofa_files()

df = hf.generate_table_difference_hrtfs(hrtf1,hrtf2)
vis.plot_LSD_left_and_right(df)
vis.plot_ild_itd_difference(df)
kaja = []