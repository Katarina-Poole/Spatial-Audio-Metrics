from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import lap_challenge as lap
from spatialaudiometrics import visualisation as vis
from spatialaudiometrics import signal_processing as sp

hrtf1 = ld.HRTF('C:/Users/katar/Box/2023_HRTF-Kathi/Data/P0124/P0124_Windowed_48kHz.sofa')


vis.plot_itd_overview(hrtf1)
vis.plot_ild_overview(hrtf1)
kaja = []