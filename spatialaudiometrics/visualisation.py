'''
Functions for visualising data
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    '''Plots errors bars using standard error and mean

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
    plt.show()
    
