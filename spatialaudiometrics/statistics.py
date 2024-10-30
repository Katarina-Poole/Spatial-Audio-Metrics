'''
Functions to do some general statistics
'''
import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import scipy.stats as sts
import scikit_posthocs as ph

def test_normality(df:pd.DataFrame,dep_col:str,subject_col:str,ind_col:list):
    '''
    Uses the shapiro test to test if normal

    :param df: Pandas dataframe that contains the data to run the stats on
    :param dep_col: dependent column name (e.g. the response variable such as polar error)
    :param subject_col: name of the column containing subject ID 
    :param ind_col: list of independent column name/s (e.g. the variable you are grouping with such as hrtf type)
    :returns bool: Returns whether the data is normal (True) or not (False)
    '''

    # Then test for normality
    if len(df[subject_col].unique()) < 50:
        # If sample size is less than 50 then do Shapiro Wilk
        # Want to run it on each indepdent variable as one independent variable may be non normal to others
        normality = True # if data is normal
        grouped = df.groupby(ind_col)
        for g, group in grouped:
            norm_stat, norm_pval = sts.shapiro(group[dep_col])
            if norm_pval < 0.05:
                print('Found non-normality')
                normality = False
    else:
        print("Sample size too big so can't use this test for normality")
        normality = None
    return normality

def tukey_hsd(df:pd.DataFrame, dep_col:str, ind_col:list):
    '''
    Creates a pariwise comparison table using the Tukey test (for normal data)

    :param df: data frame you want to run the test on (already aggregated)
    :param dep_col: dependent column name (e.g. the response variable such as polar error)
    :param ind_col: list of independent column name/s (e.g. the variable you are grouping with such as hrtf type)
    '''
    # Create groups to run the pairwise
    grouped = df.groupby(ind_col)
    pairwisegroups = list()
    pairwisegroupnames = list()
    for g, group in grouped:
        pairwisegroupnames.append(ind_col[0] + ' ' + str(g))
        pairwisegroups.append(group[dep_col].values)

    # Run Tukey test
    tukey = sts.tukey_hsd(*pairwisegroups) # takes in list and unpacks it
    tukey_stats_table = pd.DataFrame()

    # Convert to table for easy reading and getting rid of duplicates
    for i in range(np.shape(tukey.statistic)[0]): # Rows
        for j in range(np.shape(tukey.statistic)[1]):
            if i<j: # Only care about unique comparisons
                temp = pd.DataFrame([[i,j,pairwisegroupnames[i],pairwisegroupnames[j],
                                      tukey.statistic[i,j],np.round(tukey.pvalue[i,j],6)]],
                                      columns = ['idx_1','idx_2','var_1','var_2','tukey_stat','p_val'])
                tukey_stats_table = pd.concat([tukey_stats_table,temp])
    tukey_stats_table = tukey_stats_table.reset_index()

    return tukey_stats_table

def repeated_measures_anova(df:pd.DataFrame,dep_col:str,subject_col:str,ind_col:list):
    '''
    Runs a repeated measures ANOVA 

    :param df: Pandas dataframe that contains the data to run the stats on
    :param dep_col: dependent column name (e.g. the response variable such as polar error)
    :param subject_col: name of the column containing subject ID 
    :param ind_col: list of independent column name/s (e.g. the variable you are grouping with such as hrtf type)
    '''

    # Need to aggregate the data first 
    df = df.groupby(ind_col + [subject_col]).median(numeric_only = True).reset_index()

    # Then test for normality
    normality = test_normality(df,dep_col,subject_col,ind_col)

    if normality:
        results = AnovaRM(data=df, depvar=dep_col,subject=subject_col, within=ind_col).fit()
        print(results) 
        pval = results.anova_table['Pr > F'].values[0]

        if pval < 0.05:
            print('Found significant differences between groups, running pairwise comparisons as well...')
            in_text = ['was'] + [dep_col]
            # Run Tukey pairwise
            tukey_stats_table = tukey_hsd(df,dep_col,ind_col)
        else:
            in_text = ['was not'] + [dep_col]
            tukey_stats_table = []

        # Then just print the wording to use for easy copy pasting
        reporting_text = 'A one-way repeated measures ANOVA revealed that there ' + in_text[0] +  ' a statistically significant difference in ' + in_text[1] + ' between at least two groups (F(' + str(int(np.round(results.anova_table['Num DF'].values[0]))) + ',' + str(int(np.round(results.anova_table['Den DF'].values[0]))) + ') = ' + str(np.round(results.anova_table['F Value'].values[0],2)) + ', p = ' + str(np.round(pval,3)) + ')'
        
        print(reporting_text)
        print(tukey_stats_table)
        return pval, tukey_stats_table, results.anova_table
    else:
        print('Non normal, use a non-parametric test like the Friedman instead')
        return None
    
def run_friedman_test(df:pd.DataFrame,dep_col:str,subject_col:str,ind_col:list):
    '''
    Compare the mean between three or more groups if the distributions are non normal using Friedman test and Wilcoxon for post hoc pairwise

    :param df: Pandas dataframe that contains the data to run the stats on
    :param dep_col: dependent column name (e.g. the response variable such as polar error)
    :param subject_col: name of the column containing subject ID 
    :param ind_col: list of independent column name/s (e.g. the variable you are grouping with such as hrtf type)
    '''

    # Need to aggregate the data first. Again check if this is the appropriate metric!
    df = df.groupby(ind_col + [subject_col]).median(numeric_only = True).reset_index()
    grouped = df.groupby(ind_col)
    fried_groups = list()
    fried_group_names = list()
    for g, group in grouped:
        fried_groups.append(group[dep_col].values)
        fried_group_names.append(ind_col[0] + ' ' + str(g))
    
    stats = sts.friedmanchisquare(*fried_groups)
    pval = stats[1]
    if pval < 0.05:
        in_text = ['was'] + [dep_col]
        # Run pairwise comparisons using wilcoxon signed rank test
        wilcoxon_stats = ph.posthoc_wilcoxon(df,dep_col,ind_col[0],p_adjust= 'holm')
        # Want to convert the array into a table for easy reading
        wilcoxon_table = pd.DataFrame()
        for g1,group1 in enumerate(fried_groups):
            for g2, group2 in enumerate(fried_groups):
                if g1<g2:
                    temp = pd.DataFrame([[g1,g2,fried_group_names[g1],fried_group_names[g2],
                                          np.nan,wilcoxon_stats.iloc[g1,g2]]],columns = ['idx_1','idx_2','var_1','var_2','wilcoxon_stat','p_val'])
                    wilcoxon_table = pd.concat([wilcoxon_table,temp])

        wilcoxon_table = wilcoxon_table.reset_index()
    else:
        in_text = ['was not'] + [dep_col]
        wilcoxon_table = []

    reporting_text = ['A Friedman analysis of variance revealed that there ' + in_text[0] +  ' a statistically significant difference in ' + in_text[1] + ' between at least two groups (Fr = ' + str(np.round(stats[0],2)) + ', df = ' + str(int(len(fried_groups)-1)) + ', p = ' + str(np.round(pval,3)) + ')']
    
    print(reporting_text[0])
    print(wilcoxon_table)
 
    return stats,wilcoxon_table
