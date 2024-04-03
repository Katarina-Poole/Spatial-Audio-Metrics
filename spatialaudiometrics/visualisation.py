'''
Functions for visualising data
'''
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spatialaudiometrics import hrtf_metrics as hf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import ScalarMappable

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
    av      = df.groupby(x_name).mean(numeric_only = True).reset_index()
    std     = df.groupby(x_name).std(numeric_only = True).reset_index()/(len(df[sample_name].unique())**0.5)
    y       = av[y_name]
    yerr    = std[y_name]
    axes.errorbar(x, y, yerr, fmt='o', color = [0.3,0.3,0.3], linewidth = 3, markersize = 7)

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

def plot_tf_overview(hrtf,az = [0,90,180,270]):
    '''
    A function that displays the transfer function at multiple elevation locations and select azimuth locations (max four)
    
    :param hrtf: The hrtf object as loaded in by hrtf = load_data.HRTF(sofa_path)
    :param az: The azimuth locations in which the transfer functions will be plotted (max four for nice plots). Default is ahead, left, behind and right
    '''
    tfs,freqs,phase = hf.hrir2hrtf(hrtf.hrir,hrtf.fs)
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