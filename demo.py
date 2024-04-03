from spatialaudiometrics import load_data as ld
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import lap_challenge as lap
from spatialaudiometrics import visualisation as vis

# Load in some example HRTFs
hrtf1,hrtf2 = ld.load_example_sofa_files()


# Lets try some visualisation
vis.plot_hrtf_overview(hrtf1)

# If you want to load in your own you can use
#hrtf1 = ld.HRTF()
#hrtf1.load(sofa_path)

# Then we need to make sure they are matched in location positions if we want to run direct comparisons using the following code. sofa 1 should be the location of the target locations you want to keep
hrtf1,hrtf2 = ld.match_hrtf_locations(hrtf1,hrtf2)

# To calculate the itd of an hrtf we can do:
itd_s,itd_samps,maxiacc = hf.itd_estimator_maxiacce(hrtf1.hrir,hrtf1.fs)

# To calculate the difference between itd values we can do:
itd_diff = hf.calculate_itd_difference(hrtf1,hrtf2)

# To calculate the ild of an hrtf we can do:
ild = hf.ild_estimator_rms(hrtf1.hrir)

# To calculate the difference between ild values we can do:
ild_diff = hf.calculate_ild_difference(hrtf1,hrtf2)

# To calculate the LSD across locations we can do:
lsd,lsd_mat = hf.calculate_lsd_across_locations(hrtf1.hrir,hrtf2.hrir,hrtf1.fs)

# To calculate all the metrics for the LAP challenge and see if its below threshold for each metric
metrics,threshold_bool,df = lap.calculate_lap_challenge_metrics('C:\GithubRepos\Spatial-Audio-Metrics\spatialaudiometrics\example_sofa_1.sofa','C:\GithubRepos\Spatial-Audio-Metrics\spatialaudiometrics\example_sofa_2.sofa')