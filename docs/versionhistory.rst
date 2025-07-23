Version history
=================================================
- (18/10/2024) **0.1.2** Fixed project dependencies
- (18/10/2024) **0.1.0** Major release of Spatial Audio Metrics. See patch notes below:
    - Updated documentation to include tutorials on HRTF analysis and localisation peformance analysis
    - Changed all power calculations to use np.power rather than '**' due to errors using '**'
    - Added ITD property to the HRTF object in load_data and it would pull ITD from the meta data of the SOFA file if present
    - Changed pol_abs_accuracy and precision in calculate_localisation_error to just use np.std rather than circularstd (as its not necessary when looking at absolulte values)
    - Changed the way that plot_error_bar deals with the data on the y axis to make it work better across different variable types
    - Changed direction of ITD and ILD polar plot in plot_itd_overview and plot_ild_overview

    - Added functions:
        - calculate_lsd_across_locations_per_frequency
        - itd_estimator_threshold
        - generate_table_difference_hrtfs
        - generate_table_difference_lsd_freq_hrtfs
        - create_wavelet
        - wavelet_decomposition
        - plot_hrir_both_ears
        - plot_hrtf_both_ears
        - plot_spectrogram
        - plot_spectrum
        - plot_ild_itd_difference
        - plot_LSD_left_and_right
        - plot_LSD_left_right_frequency
        - plot_raw_Localisation
        - create_raw_localisation_legend
        - load_sonicom_sofa
        - plot_confusion_sphere

    - Added classes:
        - visualisation.Colours
        - visualisation.Palettes

    - Added parameters to functions:
        - itd_estimator_maxiacce
        - calculate_ild_difference

- (20/06/2024) **0.0.8** Updated LAP challenge ITD threshold
- (17/06/2024) **0.0.7** Bug fix - Updated itdestimator maxiacce to first low pass the hrir at 3000Hz with filter order of 10 as previously was just on broadband hrir
- (05/06/2024) **0.0.6** Updated LAP challenge ITD threshold
- (05/04/2024) **0.0.4** Added in source location visualisation and updated LAP challenge thresholds
- (03/04/2024) **0.0.3** Added HRTF visualisation (spectra, ITD and ILD), db2mag, changes to LAP challenge metric calculations and a tutorial to the documentation
- (22/03/2024) **0.0.2** Fixes to example sofa data
- (20/03/2024) **0.0.1** Initial upload
