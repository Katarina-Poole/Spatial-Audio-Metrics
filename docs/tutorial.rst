Tutorial
=================================================

This package can be used to visualise HRTFs and assess differences between HRTFs (currently those that are from the same database, i.e. have the same source localisations and IR length).

Loading in HRTFs
----------------------

HRTFs are typically stored as sofa files, so the the module creates an HRTF object by pulling out relevant data from the sofa file when initialising the object. You can load your own sofa file like so:

.. code-block:: python

    from spatialaudiometrics import load_data as ld

    hrtf1 = ld.HRTF(hrtf_original)
    hrtf2 = ld.HRTF(hrtf_upsampled)

There are also two example sofa files you can load in:

.. code-block:: python

    hrtf1,hrtf2 = ld.load_example_sofa_files()

After loading in the HRTFs, to run any direct comparisons we need to make sure the source locations are matched between them and reorder them if not. This can be done like so:

.. code-block:: python

    hrtf1,hrtf2 = ld.match_hrtf_locations(hrtf1,hrtf2)

Calculating HRTF metrics
-----------------------------------------
The function hrtf_metrics allows us to calculate a number of metrics regarding an HRTF
For example we can calculate the ILD and ITD at each location as well as pull out the transfer function (the spectra).

.. code-block:: python

    from spatialaudiometrics import hrtf_metrics as hf

    spectra, freqs, phase   = hrir2hrtf(hrtf1.hrir,hrtf1.fs)
    ild                     = ild_estimator_rms(hrtf1.hrir)
    itd_s,itd_samps,maxiacc = itd_estimator_maxiacce(hrtf1.hrir,fs)


Visualising HRTF metrics
-----------------------------------------
The function visualisation can be used to easily visualise some basic HRTF metrics such as the ILD, ITD and spectra and the source locations. The az argument in plot_hrtf_overview is used to select which azimuth locations you want to visualise the transfer functions (max four)

.. code-block:: python

    from spatialaudiometrics import visualisation as vis

    vis.plot_itd_overview(hrtf1)
    vis.plot_ild_overview(hrtf1)
    vis.plot_tf_overview(hrtf1,az = [0,90,180,270])
    vis.plot_source_locations(hrtf1.locs)

Calculating differences between HRTFs
-----------------------------------------

Then to calculate differences between the hrtfs we can use the following functions:

.. code-block:: python

    from spatialaudiometrics import hrtf_metrics as hf

    itd_diff = hf.calculate_itd_difference(hrtf1,hrtf2)
    ild_diff = hf.calculate_ild_difference(hrtf1,hrtf2)
    lsd,lsd_mat = hf.calculate_lsd_across_locations(hrtf1.hrir,hrtf2.hrir,hrtf1.fs)

This will give you single values (averaged across source positions and/or frequency).