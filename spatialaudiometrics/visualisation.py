'''
Functions for visualising data
'''
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from spatialaudiometrics import hrtf_metrics as hf
from spatialaudiometrics import angular_metrics as am
from spatialaudiometrics import signal_processing as sp


class FigSize:
    '''
    Size of figure
    '''
    # optimimum size for thesis and gives a bit of space at the bottom for the beginning of a figure legend
    # Using this for now until i find better heights and widths 
    fig_width   = 12
    fig_height  = 8

class SubplotSpacing:
    '''
    This spacing gives minimal white space round edges (but gives space for panel labels)
    '''
    top         = 0.95
    bottom      = 0.05
    left        = 0.05
    right       = 0.95

class FontSize:
    '''
    Default font size I want
    '''
    panel_fs        = 16
    legend_title    = 10
    legend          = 9
    ticks           = 8
    axis_labels     = 9
    default         = 10
    fig_title       = 12

class Colours:
    dict = {  
        'L':colors.hex2color('#196AA5'),
        'R':colors.hex2color('#C75E6B')
        }

class Palettes:
    '''
    Custom palettes for use in seaborn plotting
    '''
    left_right = sns.color_palette([Colours.dict['L'],Colours.dict['R']])
    confusion = {"precision":"black","front-back":"red","in-cone":"blue","off-cone":"orange"}

def create_fig(fig_size = (FigSize.fig_width,FigSize.fig_height),grid_spec_rows = 12, grid_spec_cols = 12):
    '''
    Creates a figure panel and then splits it into a 12x12 grid to use for subplotting

    :param fig_size: a tuple of fig width and fig height
    :param grid_spec_rows: if don't want 12 rows separation then change this to another integer
    :param grid_spec_cols: if you dont want 12 cols separation then change this to another integer
    '''
    fig     = plt.figure(figsize = fig_size)
    gs      = matplotlib.gridspec.GridSpec(grid_spec_rows,grid_spec_cols, figure=fig, hspace = 0.6, wspace = 1, top = SubplotSpacing.top, 
    bottom  = SubplotSpacing.bottom,left = SubplotSpacing.left, right = SubplotSpacing.right) # Just doing it the 12 column way as that seems the easiest and is used in dash

    plt.rc('font', size=FontSize.default)          # controls default text sizes
    plt.rc('axes', titlesize=FontSize.axis_labels)     # fontsize of the axes title
    plt.rc('axes', labelsize=FontSize.axis_labels)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FontSize.ticks)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FontSize.ticks)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FontSize.legend)    # legend fontsize
    plt.rc('figure', titlesize=FontSize.fig_title)  # fontsize of the figure title

    return fig,gs

def finish_axes(axes,grid = 1,legend = 0):
    ''' 
    Goes through and makes cosmetic adjustments to the axis

    :param axes: axes to change
    :param grid: To add grid lines (0 = no grid line, 1 = y axis lines [default], 2 = x and y axis lines)
    '''
    plt.sca(axes)
    axes.spines['top'].set_visible(False); 
    axes.spines['right'].set_visible(False)
    if grid == 1:
        axes.grid(axis = 'y', color = [0.9,0.9,0.9])
        axes.set_axisbelow(True)
    elif grid == 2:
        axes.grid(axis = 'y', color = [0.9,0.9,0.9])
        axes.grid(axis = 'x', color = [0.9,0.9,0.9])
        axes.set_axisbelow(True)
    if legend == 1:
        plt.setp(axes.get_legend().get_texts(), fontsize=FontSize.legend) # for legend text
        plt.setp(axes.get_legend().get_title(), fontsize=FontSize.legend_title) # for legend title
    else:
        plt.legend([],[], frameon=False)
    
def plot_sig_bar(axes,p_val):
    '''
    Adds a significant bar if there is a sigificant p_value

    :param axes: axes you want to plot onto
    :param p_val: the p value
    '''
    ylims = axes.get_ylim()
    y_coord_count = 0

    # Go through the pairwise comparison table and find if there are any significant pairwise comparisons
    if p_val < 0.05:
        y_coord = np.round(np.diff(ylims)*(0.05*y_coord_count) + ylims[1])[0] # 5% of the axes range
        axes.plot([0,1],[y_coord,y_coord],'k')
        x_coord = 0.5
        if p_val < 0.001:
            axes.annotate('***',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')
        elif p_val < 0.01:
            axes.annotate('**',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')
        elif p_val < 0.05:
            axes.annotate('*',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')

def plot_sig_bars_pairwise(axes,stats):
    '''
    Plots the significance bars of pairwise comparisons

    :param axes: the axes you want to plot hte significance bars on
    :param stats: The pairwise comparison table from the statistics module
    '''

    ylims = axes.get_ylim()
    y_coord_count = 0

    # Go through the pairwise comparison table and find if there are any significant pairwise comparisons
    try:
        for s,stat in stats.iterrows():
            pval = stat.p_val # Usually the last value
            if pval < 0.05:
                y_coord = (np.diff(ylims)*(0.05*y_coord_count) + ylims[1])[0] # 5% of the axes range
                axes.plot([stat.idx_1,stat.idx_2],[y_coord,y_coord],'k')
                x_coord = ((stat.idx_2-stat.idx_1)/2) + stat.idx_1
                if pval < 0.001:
                    axes.annotate('***',[x_coord,y_coord*1.0005],fontsize=14,ha='center')
                elif pval < 0.01:
                    axes.annotate('**',[x_coord,y_coord*1.0005],fontsize=14,ha='center')
                elif pval < 0.05:
                    axes.annotate('*',[x_coord,y_coord*1.0005],fontsize=14,ha='center')
                y_coord_count +=1

    except:
        print('No pairwise comparisons to plot')

def plot_error_bar(axes,df,x_name,y_name,sample_name):
    '''
    Plots errors bars using standard error and mean

    :param axes: The axes you want to plot in
    :param df: The dataframe you want to use for the error bar
    :param x_name: The variable on the x axis
    :param y_name: The variable on the y axis
    :param sample_name: The name of the column giving the subject or sample identifier
    '''

    x       = np.arange(0,len(df[x_name].unique()),1) + 0.2
    y       = df.groupby(x_name).mean(numeric_only = True).reset_index()[y_name]
    std     = df.groupby(x_name).std(numeric_only = True).reset_index()[y_name]/(len(df[sample_name].unique())**0.5)
    axes.errorbar(x, y, std, fmt='o', color = [0.3,0.3,0.3], linewidth = 3, markersize = 7)

def annotate_p_vals(axes,pval,x_coord):
    '''
    Adds p value indicators to the plot

    :param axes: axes to plot onto
    :param pval: the p value
    :param x_coord: the x coordinate of the plot to annotate at
    '''
    ylims           = axes.get_ylim()
    y_coord_count   = 0
    if pval < 0.05:
        y_coord = np.round(np.diff(ylims)*(0.05*y_coord_count) + ylims[1])[0] # 5% of the axes range
    if pval < 0.001:
        axes.annotate('***',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')
    elif pval < 0.01:
        axes.annotate('**',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')
    elif pval < 0.05:
        axes.annotate('*',[x_coord,y_coord*1.001],fontsize=FontSize.panel_fs,ha='center')
    y_coord_count +=1

def plot_model_table(model,axes):
    '''
    A plot to show the coefficients and confidence intervals of each predictor of a glm

    :param model: Model object from 
    '''
    confidence_intervals    = model.conf_int(alpha=0.05)
    p_values                = model.pvalues
    x                       = np.arange(0,len(model.params),1)
    y                       = list()
    err                     = list()
    for i in range(len(model.params)):
        y.append(model.params.iloc[i])
        err.append(confidence_intervals.iloc[i,0])
        err.append(confidence_intervals.iloc[i,1])
        err.append(np.nan)

    # Plot the confidence intervals
    axes.plot(err,np.repeat(x,3),'-',linewidth = 15, color = [.8,.8,.8])
    # Plot the sig predictors
    for i,p in enumerate(p_values):
        if p < 0.05:
            plt.plot(y[i],x[i],'.', markersize = 18, color = 'k')
        else:
            plt.plot(y[i],x[i],'.', markersize = 18, color = 'k', markerfacecolor = 'white')

    axes.set_yticks(x)
    axes.set_yticklabels(model.params.keys())
    axes.set_ylim(-1,len(x))
    axes.vlines(0,ymin = -1, ymax = len(x),linestyle = '--', color = 'grey', linewidth = 1)
    axes.set_ylabel('Predictor')
    axes.set_xlabel('Coefficient')
    finish_axes(axes)

def show():
    '''
    To show the plots
    '''
    plt.show(block = False)

def plot_tf_overview(hrtf,type = 'hrtf',az = [0,90,180,270]):
    '''
    A function that displays the transfer function at multiple elevation locations and select azimuth locations (max four)
    
    :param hrtf: The hrtf object as loaded in by hrtf = load_data.HRTF(sofa_path)
    :param az: The azimuth locations in which the transfer functions will be plotted (max four for nice plots). Default is ahead, left, behind and right
    '''
    if type == 'hrtf':
        tfs,freqs,phase = hf.hrir2hrtf(hrtf.hrir,hrtf.fs)
    elif type == 'dtf':
        tfs,freqs,phase = hf.hrir2hrtf(hrtf.dtf_ir,hrtf.fs)

    freq_idx        = np.where((freqs >= 20) & (freqs <=20000))[0]
    freqs           = freqs[freq_idx]
    # Get the color map limits
    vmin            = np.floor(np.min(tfs[np.in1d(hrtf.locs[:,0],az),:,:]))
    vmax            = np.ceil(np.max(tfs[np.in1d(hrtf.locs[:,0],az),:,:]))

    fig_titles      = ['Left ear',' Right ear']
    for e in range(2):
        fig,gs      = create_fig(fig_size=(12,8))
        subplots    = [gs[0:5,1:5],gs[0:5,7:11],gs[6:11,1:5],gs[6:11,7:11]]

        for i,curr_az in enumerate(az):
            axes            = fig.add_subplot(subplots[i])
            az_loc_idx      = np.where(hrtf.locs[:,0] == curr_az)[0]
            el_angles       = np.squeeze(hrtf.locs[az_loc_idx,1])
            sort_idx        = np.argsort(el_angles)
            el_angles       = el_angles[sort_idx]
            curr_tf         = np.squeeze(tfs[az_loc_idx,e,:])
            curr_tf         = curr_tf[sort_idx,:]
            c               = axes.contourf(curr_tf[:,freq_idx],levels = 100,cmap = 'magma',vmin = vmin, vmax = vmax)
            cax             = inset_axes(axes,width="2.5%",height="100%",bbox_transform=axes.transAxes,bbox_to_anchor=(0.025, 0.05, 1.05, 0.95),loc= 1)
            
            plt.colorbar(c,cax = cax)
            axes.set_ylabel('Elevation (°)')
            axes.set_yticks(range(len(el_angles)))
            axes.set_yticklabels(el_angles)
            axes.set_xlabel('Frequency (kHz)')
            axes.set_xticks(np.arange(0,len(freqs),15))
            axes.set_xticklabels(np.round(freqs[np.arange(0,len(freqs),15)]/1000,2))
            axes.set_xlim(freq_idx[0],freq_idx[-1])
            cax.set_ylabel('dB')
            axes.set_title('Azimuth: ' + str(curr_az) + '°')
            fig.suptitle(fig_titles[e])
    show()

def plot_itd_overview(hrtf):
    '''
    Plots the ITD at elevation 0 and also the itd by each location
    '''
    itd_s,itd_samps,maxiacc = hf.itd_estimator_maxiacce(hrtf.hrir,hrtf.fs)
    itd_us                  = itd_s * 1000000
    idx                     = np.where(hrtf.locs[:,1] == 0)[0]
    sort_idx                = np.argsort(hrtf.locs[idx,0])
    idx                     = idx[sort_idx]
    
    fig,gs  = create_fig(fig_size=(16,6))
    axes    = fig.add_subplot(gs[0:12,0:4], projection = 'polar')
    axes.set_theta_direction(-1)
    axes.plot(np.deg2rad(hrtf.locs[idx,0]),np.abs(itd_us[idx]))
    axes.set_theta_zero_location("N")
    axes.set_rticks([200,400,600,800])
    axes.set_title('Absolute ITD (µs)')

    axes    = fig.add_subplot(gs[2:10,5:12])
    huemax  = max(abs(itd_us))
    sns.scatterplot(x = hrtf.locs[:,0], y = hrtf.locs[:,1], hue = itd_us, hue_norm=(-huemax,huemax), ax = axes, palette = "vlag")
    axes.set_ylabel('Elevation (°)')
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    axes.set_title('ITD at all locations')
    finish_axes(axes)
    add_colourbar_on_side(-huemax,huemax,"vlag", axes, 'ITD (µs)')
    show()

def plot_ild_overview(hrtf):
    '''
    Plots the ILD at elevation 0 and also the ild by each location
    '''
    ild                     = hf.ild_estimator_rms(hrtf.hrir)
    idx                     = np.where(hrtf.locs[:,1] == 0)[0]
    sort_idx                = np.argsort(hrtf.locs[idx,0])
    idx                     = idx[sort_idx]

    fig,gs  = create_fig(fig_size=(16,6))
    axes    = fig.add_subplot(gs[0:12,0:4], projection = 'polar')
    axes.set_theta_direction(-1)
    axes.plot(np.deg2rad(hrtf.locs[idx,0]),np.abs(ild[idx]))
    axes.set_theta_zero_location("N")
    #axes.set_rticks([200,400,600,800])
    axes.set_title('Absolute ILD (dB)')

    axes    = fig.add_subplot(gs[2:10,5:12])
    huemax  = max(abs(ild))
    sns.scatterplot(x = hrtf.locs[:,0], y = hrtf.locs[:,1], hue = ild, hue_norm=(-huemax,huemax), ax = axes, palette = "vlag")
    axes.set_ylabel('Elevation (°)')
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    axes.set_title('ILD at all locations')
    finish_axes(axes)
    add_colourbar_on_side(-huemax,huemax,"vlag", axes, 'ILD (dB)')
    show()

def add_colourbar_on_side(hue_min,hue_max,colourmap,axes,axes_label):
    '''
    Adds a colourbar on the side of the plot
    
    :param hue_min: the minimum hue value used
    :param hu_max: the maximum hie value used
    :param colourmap: the colourmap used
    :param axes: the axes you want to generate it on the side of
    :param axes_label: the label you want to attach to the colourbar     
    '''
    norm    = plt.Normalize(hue_min,hue_max)
    sm      = plt.cm.ScalarMappable(cmap = colourmap, norm = norm)
    sm.set_array([])
    cbar    = axes.figure.colorbar(sm, ax = axes)
    cbar.set_label(axes_label, rotation = 90)

def plot_source_locations(locs,elevation_palette = "tab10", point_size = 25):
    '''
    Plots the source locations on a 3d plot.
    
    :param locations: numpy array where each row is a location, col 1 = azimuth, col 2 = elevation, col 3 = distance. Can easily just use hrtf.locs
    '''
    # Plot the locations used
    fig,gs = create_fig(fig_size=(5,5))
    x,y,z = am.polar2cartesian(locs[:,0], locs[:,1], locs[:,2])
    dist = max(abs(z))
    axes = fig.add_subplot(gs[0:12,0:12],projection='3d')
    axes.scatter(x,y,z,s = point_size,c = z, cmap = elevation_palette)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Source locations')
    axes.set_ylim(-dist,dist)
    axes.set_xlim(-dist,dist)
    axes.set_zlim(-dist,dist)
    axes.set_aspect('equal')
    # Annotate
    x,y,z = am.polar2cartesian(0, 0, locs[0,2])
    axes.text(x,y,z,  'Front', size=14, color = 'k',)
    x,y,z = am.polar2cartesian(90, 0, locs[0,2])
    axes.text(x,y,z,  'Left', size=14, color = 'k')
    x,y,z = am.polar2cartesian(270, 0, locs[0,2])
    axes.text(x,y,z,  'Right', size=14, color = 'k')
    finish_axes(axes)
    show()
    return fig, axes

def create_source_location_gif(fig,axes,save_filename,dpi = 120):
    '''
    Creates a gif that rotates around the azimuth 
    
    :param fig: matplotlib figure handles
    :param axes: matplotlib ax handle you want to rotate
    :param save_filename: filename including path of where you want to save the file (include .gif)
    :param dpi: dpi settings for animation, higher value = better quality but longer rendering time
    '''
    def rotate(angle):
        axes.view_init(azim=angle)

    print("Making animation...")
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    rot_animation.save(save_filename, dpi=dpi, writer='pillow')
    print("Saved animation at: " + save_filename)


def plot_hrir_both_ears(hrtf,az,el,axes):
    '''
    Plots the hrir at a given position for left and right ears
    '''
    # Get location
    idx = np.where((hrtf.locs[:,0] == az) & (hrtf.locs[:,1] == el))[0][0]

    hrir_l = hrtf.hrir[idx,0,:]
    hrir_r = hrtf.hrir[idx,1,:]

    ts = (np.arange(0,len(hrir_l))/hrtf.fs)*1000
    axes.plot(ts,hrir_l,color = Colours.dict['L'])
    axes.plot(ts,hrir_r,color = Colours.dict['R'])
    axes.set_ylabel('Amplitude')
    axes.set_xlabel('Time (ms)')
    axes.set_title('Azimuth: ' + str(az) + '°, elevation: ' + str(el) + '°.')
    finish_axes(axes)

def plot_hrtf_both_ears(hrtf,az,el,axes,log_scale = True):
    '''
    lots the transfer function at a given position for left and right ears
    '''
    idx = np.where((hrtf.locs[:,0] == az) & (hrtf.locs[:,1] == el))[0][0]

    hrtfs, freqs, hrtfs_phase = hf.hrir2hrtf(hrtf.hrir,hrtf.fs,db_flag = 1)

    hrtf_l = hrtfs[idx,0,:]
    hrtf_r = hrtfs[idx,1,:]

    axes.plot(freqs/1000,hrtf_l,color = Colours.dict['L'])
    axes.plot(freqs/1000,hrtf_r,color = Colours.dict['R'])
    axes.set_ylabel('Magnitude (dB)')
    axes.set_xlabel('Frequency (kHz)')
    axes.set_title('Azimuth: ' + str(az) + '°, elevation: ' + str(el) + '°.')
    if log_scale:
        axes.set_xscale('log')
        axes.set_xticks([0.125,0.25,0.5,1,2,4,8,16,22])
        axes.set_xticklabels([0.125,0.25,0.5,1,2,4,8,16,22])

    finish_axes(axes)

def plot_spectrogram(sig,fs,axes,vmin_max = False, freq_max = 20000, db_flag = 0):
    '''
    Plots the spectrogram of a signal using wavelet decomposition
    :param sig: 1D signal you want to run the spectrogram on
    :param fs: sample rate
    :param axes: axes you want to plot the spectrogram on
    :vmin_max: a tuple with the min and max values you want to display
    :param freq_max: the maximum frequency you want to calculate for
    :param db_flag: if you want the spectrogram to display in magnitude (0) or dB (1)
    :return freqs: frequencies
    :return mag: magnitude or dB of signal at each frequency
    :return phase: phase of the signal at each frequency
    '''
    #print('Running wavelet decomposition...')
    #mag,phase,freqs = sp.wavelet_decomposition(sig,fs,freq_steps = 10, freq_max = freq_max)

    # Just using the scipy one for ow
    freqs, ts, mag = scipy.signal.spectrogram(sig, fs)

    if db_flag == 1:
        mag = sp.mag2db(mag)

    print('Plotting...')
    #ts = np.arange(0,len(sig)/fs,1/fs)
    if vmin_max:
        c = axes.pcolorfast(ts,freqs/1000,mag,cmap = 'mako',vmin = vmin_max[0],vmax = vmin_max[1])
    else:
        c = axes.pcolorfast(ts,freqs/1000,mag,cmap = 'mako')

    axes.set_xlabel('Time (s)') 
    axes.set_ylabel('Frequency (kHz)')
    #axes.set_yscale('log')
    cax = inset_axes(axes,width="2.5%",height="100%",bbox_transform=axes.transAxes,bbox_to_anchor=(0.025, 0.05, 1.05, 0.95),loc= 1)
    plt.colorbar(c,cax = cax)

    return freqs,mag  #phase


def plot_spectrum(sig,fs,axes):
    '''
    Plots the spectrum
    '''
    spec, freqs, phase = sp.calculate_spectrum(sig,fs)
    axes.plot(freqs/1000,spec,'k')
    axes.set_xlabel('Frequency (kHz)')
    axes.set_ylabel('Magnitude (dB)')
    finish_axes(axes)

def plot_ild_itd_difference(df,diverging = True):
    '''
    Creates a plot to show the itd and ild difference
    '''
    # Plot ITD difference
    fig,gs = create_fig()
    axes = fig.add_subplot(gs[1:6,1:12])
    if diverging:
        palette = "vlag"
        huemax = max(abs(df.itd_diff_us))
        huemin = huemax
    else:
        palette = "viridis"
        huemin = min(df.itd_diff_us)
        huemax = max(df.itd_diff_us)

    sns.scatterplot(data=df, x="az", y="el", hue = "itd_diff_us", hue_norm = (-huemin,huemax), ax = axes, palette = palette)
    axes.set_title('ITD difference')
    axes.set_ylabel('Elevation (°)')
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    finish_axes(axes)
    add_colourbar_on_side(-huemin,huemax,palette,axes,'ITD difference (μs)')

    # Plot ILD difference
    if diverging:
        huemax = max(abs(df.ild_diff_db))
        huemin = huemax
    else:
        huemin = min(df.ild_diff_db)
        huemax = max(df.ild_diff_db)

    axes = fig.add_subplot(gs[7:12,1:12])
    sns.scatterplot(data=df, x="az", y="el", hue = "ild_diff_db", hue_norm = (-huemin,huemax), ax = axes, palette = palette)
    axes.set_title('ILD difference')
    axes.set_ylabel('Elevation (°)')
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    finish_axes(axes)
    add_colourbar_on_side(-huemin,huemax,palette,axes,'ILD difference (dB)')
    show()
    return fig

def plot_LSD_left_and_right(df):
    '''
    Plot on the same plot of the log spectral distortion on the L and R given a data frame with it already generated
    '''
    fig,gs = create_fig()
    maxLSD = max([max(abs(df.lsd_l)),max(abs(df.lsd_r))])
    minLSD = min([min(abs(df.lsd_l)),min(abs(df.lsd_r))])
    axes = fig.add_subplot(gs[1:6,1:12])
    sns.scatterplot(data=df, x="az", y="el", hue_norm = (minLSD,maxLSD), hue = "lsd_l", ax = axes, palette = "PuRd")
    axes.set_title('LSD of left ear')
    axes.set_ylabel('Elevation (°)')
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    finish_axes(axes)
    add_colourbar_on_side(minLSD,maxLSD,"PuRd",axes,'Log spectral distortion (dB)')

    axes = fig.add_subplot(gs[7:12,1:12])
    sns.scatterplot(data=df, x="az", y="el", hue_norm = (minLSD,maxLSD), hue = "lsd_r", ax = axes, palette = "PuRd")
    axes.set_title('LSD of right ear')
    axes.set_ylabel('Elevation (°)') 
    axes.set_xlabel('Azimuth (°); -> counterclockwise')
    finish_axes(axes)
    add_colourbar_on_side(minLSD,maxLSD,"PuRd",axes,'Log spectral distortion (dB)')
    show()
    return fig

def plot_LSD_left_right_frequency(df):
    '''
    Plot the LSD as a function of frequency
    '''
    fig,gs = create_fig(fig_size = (8,4))
    df['freqs'] = df['freqs']/1000
    axes = sns.lineplot(data = df, x = "freqs", y = "lsd", hue = "ear",palette = Palettes.left_right, errorbar = "sd")
    axes.set_title("Log spectral distortion across location for each frequency (synthetic/measured)")
    axes.set_ylabel('Log spectral distortion (dB)')
    axes.set_xlabel('Frequency (kHz)')
    axes.set_xlim([0,24])
    axes.set_xscale('log')
    freqs = [0.25,0.5,1,2,4,8,12,16,20]
    axes.set_xticks(freqs)
    axes.set_xticklabels(freqs)
    axes.set_xlim([0.2,20])
    finish_axes(axes,legend = 1)

    return fig

def plot_raw_localisation(df,axes,coord):
    '''
    Plots a scatterplot with target vs. response
    :param df: data frame to plot - that has the coord_target and also the confusion classification
    :param axes: subplot axes to plot in
    :param coord: what coordinate do you want to plot? (e.g. azi, ele, lat, pol)
    '''
    sns.scatterplot(data = df, x = coord + '_target', y = coord + '_response',alpha = 0.5, hue = 'confusion_classification',s = 40 ,ax = axes,palette = Palettes.confusion)
    axes.set_ylabel('Response')
    axes.set_xlabel('Target')
    axes.set_box_aspect(1)
    finish_axes(axes,grid = 2)

def create_raw_localisation_legend(axes):
    # Need to do a manual legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='precision', markerfacecolor='black', markersize=10,alpha = 0.5),
                    Line2D([0], [0], marker='o', color='w', label='front-back', markerfacecolor='red', markersize=10,alpha = 0.5),
                    Line2D([0], [0], marker='o', color='w', label='in-cone', markerfacecolor='blue', markersize=10,alpha = 0.5),
                    Line2D([0], [0], marker='o', color='w', label='off-cone', markerfacecolor='orange', markersize=10,alpha = 0.5)]
    axes.legend(handles = legend_elements)

def plot_confusion_sphere(df,azi_target:float,ele_target:float, point_size = 25):
    '''
    Plots the response to a specific target in a sphere and colours by confusion
    :param df: Dataframe of the preprocessed behavioural data 
    :param azi_target: Azimuth coordinate of the target to filter by
    :param ele_taret: Elevation coordinate of the target to filter by
    :param point_size: Size of points on sphere
    '''
    df              = df.loc[((df.azi_target == azi_target) & (df.ele_target == ele_target))]
    distance        = 1.5
    df['distance']  = distance # only here for the visualisation
    fig,gs          = create_fig(fig_size=(5,5))

    x,y,z           = am.polar2cartesian(df.azi_response, df.ele_response, df.distance)
    dist            = max(abs(z))
    axes            = fig.add_subplot(gs[0:12,0:12],projection='3d')
    
    axes.scatter(x,y,z,s = point_size,c = [Palettes.confusion[i] for i in df.confusion_classification],alpha = 0.5)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Source locations')
    axes.set_ylim(-dist,dist)
    axes.set_xlim(-dist,dist)
    axes.set_zlim(-dist,dist)
    axes.set_aspect('equal')
    # Annotate
    x,y,z = am.polar2cartesian(0, 0, distance)
    axes.text(x,y,z,  'Front', size=14, color = 'k',)
    x,y,z = am.polar2cartesian(90, 0, distance)
    axes.text(x,y,z,  'Left', size=14, color = 'k')
    x,y,z = am.polar2cartesian(270, 0, distance)
    axes.text(x,y,z,  'Right', size=14, color = 'k')
    finish_axes(axes)
    create_raw_localisation_legend(axes)
    show()
    return fig, axes