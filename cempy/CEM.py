"""Main module."""
# Load the class

# CEM with Python

# Load libraries
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from matplotlib import pyplot as plt
from scipy import stats


# CEM Class definition

class CEM:

    """
    Creates a new CEM object. This is the first step to performing Coarsened Exact Machine using this package.
    ...

    Attributes
    ----------

    Methods
    -------


    """

    def __init__(self, data, trt, metric=[], matching_on=[], exclude=[]):

        """
        Parameters
        ----------
        data: DataFrame
            Pandas data frame
        trt: str
            The name of the trt variable as appears in the DataFrame
        metric: list
            List of column names that are of primary interest to perform evaluation about
        matching_on: List
            List of columns names on which the matching will be performed
        exclude: List, optional
            List of column names that are to be excluded from the analysis, but they will still be retained in the data

        """
        self.matching_on = [i for i in matching_on if i not in exclude + metric]

        # Data cleaning process
        # Remove missing values if found in matching_on

        self.data = data.dropna(subset=matching_on)
        self.trt = trt
        self.metric = metric

        # Create bin markers by concatenating the values of `matching_on`
        self.data['bin_marker'] = self.data[self.matching_on[0]].astype(str)
        for i in range(1, len(self.matching_on)):
            self.data['bin_marker'] = self.data['bin_marker'].astype(str) + '_' + self.data[self.matching_on[i]].astype(str)

        # Define trt control dataframes
        self.t = self.data.loc[self.data[trt] == 1]
        self.c = self.data.loc[self.data[trt] == 0]

        # Other attributes

        # CEM summary table
        self.ms = self.summary(self.trt)
        self.nt_matched = self.ms.groupby(['is_matched']).sum().iloc[1, 0]
        self.nc_matched = self.ms.groupby(['is_matched']).sum().iloc[1, 1]

        # CEM summary with weights for each record
        self.msw = self.get_weights()
        self.msw[self.trt] = 0
        self.mt = self.matched_table()

        # Data with weights
        self.weighted_data = self.get_weighted_data()


    # Return the list of matched variables
    def matched_var(self):

        """
        Returns a list of variables used to perform matching

        """
        return list(self.matching_on)


    # Create CEM summary

    def summary(self, treatment=False):
        """
        summary method performs many-to-many matching and returns a copy of the original `dataframe` that
        contains addition column called 'is_matched' which indicates whether a member has a match or not
        """

        # Group members by bin signature andc ount how many members belong to each type

        self.t_grouped = self.t.groupby(['bin_marker']).size().to_frame(name='trt_count').reset_index()
        self.c_grouped = self.c.groupby(['bin_marker']).size().to_frame(name='ctl_count').reset_index()

        # Combine all grouped members into a single table
        self.member_summary = pd.merge(self.t_grouped, self.c_grouped, how='outer', on=['bin_marker'])

        # Replace NAs with 0
        self.member_summary = self.member_summary.fillna(0)

        # Add indicator column for matched bin signature
        self.member_summary['is_matched'] = 0
        self.member_summary.loc[
            (self.member_summary['trt_count'] > 0) & (self.member_summary['ctl_count'] > 0), 'is_matched'] = 1

        return self.member_summary


    def get_matched(self):

        """
        Returns the matched dataframe
        """

        # 1. Matching
        self.matched_data = self.summary(self.trt)

        # Get a list of bins that are matched
        bin_matched = self.matched_data.loc[self.matched_data['is_matched'] == 1]['bin_marker']
        bin_matched = list(bin_matched)

        # Get the matched members out
        data_matched = self.data[self.data['bin_marker'].isin(bin_matched)]

        return data_matched


    def get_unmatched(self):

        """
        Returns the unmatched dataframe
        """

        # 1. Matching
        self.matched_data = self.summary(self.trt)

        # Get a list of bins that are matched
        bin_unmatched = self.matched_data.loc[self.matched_data['is_matched'] == 0]['bin_marker']
        bin_unmatched = list(bin_unmatched)

        # Get matched members out
        data_unmatched = self.data[self.data['bin_marker'].isin(bin_unmatched)]

        return data_unmatched


    def matched_table(self):
        """
        Returns a table of matched or unmatched members in the trt and control groups
        """

        summary_matched = self.summary(self.trt)[['trt_count', 'ctl_count', 'is_matched']]

        return summary_matched.groupby(['is_matched']).sum()


    def match_rate_out(self):

        """
        Calculate matching rates: overall, trt and control
        """

        overall = (self.mt.iloc[1].sum()) / (self.mt.sum().sum())
        treatment = (self.mt.iloc[1, 0].sum()) / (self.mt.iloc[:, 0].sum())
        control = (self.mt.iloc[1, 1].sum()) / (self.mt.iloc[:, 1].sum())

        return (overall, treatment, control)


    def get_weights(self):

        """
        Returns a dataframe with CEM summary along with bin_markers, treatment counts,
        control counts, a flag whether matched was found and the weight.
        """
        # Matched control members get normalized weights according to trt group
        members_summary_matched = self.ms.loc[(self.ms['is_matched'] > 0)]
        members_summary_matched['wt'] = (members_summary_matched['trt_count'] / members_summary_matched['ctl_count']) * \
                                        (self.nc_matched / self.nt_matched)

        return members_summary_matched


    def get_weighted_data(self):

        """
        Returns a dataframe with the calculated weights added as a column.
        Treatment rows will receive a wt = 1 while control will receive a calculated weight

        """
        df_weighted = self.get_weights()
        data_matched = self.get_matched()

        todrop = ['trt_count', 'ctl_count', 'is_matched']
        df_weighted = df_weighted.drop(columns=todrop)

        # Match corresponding weight for every member
        data_matched['bin_marker'] = data_matched['bin_marker'].astype(str)
        df_weighted['bin_marker'] = df_weighted['bin_marker'].astype(str)
        data_matched_w_weights = pd.merge(data_matched, df_weighted, how='left', on=['bin_marker'])

        # Treatment cases get a weight of 1
        data_matched_w_weights.loc[data_matched_w_weights[self.trt] == 1, 'wt'] = 1
        data_matched_w_weights = data_matched_w_weights.reset_index(drop=True)

        return data_matched_w_weights


    def _effect_out(self, metric):
        """
        Internal function for use within the metric_summary()
        """

        df_tmp = self.data[[self.trt, 'bin_marker', metric]]
        df_tmp = df_tmp.groupby(['bin_marker', self.trt]).sum().reset_index()
        effect = pd.merge(df_tmp, self.msw, how='left', on=['bin_marker'])
        effect = effect.fillna(0)

        # Calculate effect for the treatment and control groups
        trt_effect = ((effect[self.trt + '_x'] * effect['is_matched'] * effect[metric]).sum()) / self.nt_matched
        ctl_effect = ((1 - effect[self.trt + '_x']) * effect['is_matched'] * effect[metric] * \
                      effect['wt']).sum() / self.nc_matched

        return (trt_effect, ctl_effect)


    def metric_summary(self, metrics=False):

        """
        Returns a summary table for the metrics after matching is performed.
        Shows estimated impact of each metrics, their 95% normal confidence intervals

        """
        if not metrics:
            metrics = self.metric

        Control = 'Control'

        i = 1
        metric_output = pd.DataFrame(
            columns=[
                'Treatment'
                , 'Treatment group - Matched (n)'
                , 'Treatment group - Match rate'
                , 'Control'
                , 'Control group - Matched (n)'
                , 'Control group - Match rate'
                , 'Metric'
                , 'Metric (Treatment)'
                , 'Metric (Treatment) LCF'
                , 'Metric (Treatment) UCF'
                , 'Metric (Control)'
                , 'Metric (Control) LCF'
                , 'Metric (Control) UCF'
                , 'Metric Impact'
                , 'Metric Impact (LCF)'
                , 'Metric Impact (UCF)'
                , 'Metric Impact (%)'
                , 'Metric Impact (%) LCF'
                , 'Metric Impact (%) UCF'
                , 'p-value'
            ]
        )

        control_weights = self.weighted_data.loc[self.weighted_data[self.trt] == 0, 'wt']
        control_n = (self.weighted_data[self.trt] == 0).sum()
        treatment_n = (self.weighted_data[self.trt] == 1).sum()

        for metric in metrics:

            effect = self._effect_out(metric)

            control_metric = self.weighted_data.loc[self.weighted_data[self.trt] == 0, metric]
            control_mean = np.average(control_metric, weights=control_weights)
            control_std = np.sqrt(np.average((control_metric - control_mean) ** 2, weights=control_weights))

            if np.float(control_mean) != 0:
                control_LCI = control_mean - 1.96 * control_std / np.sqrt(control_n)
                control_UCI = control_mean + 1.96 * control_std / np.sqrt(control_n)
            else:
                control_LCI = 0.00001
                control_UCI = 0.00001

            treatment_metric = self.weighted_data.loc[self.weighted_data[self.trt] == 1, metric]
            treatment_mean = treatment_metric.mean()
            treatment_std = np.sqrt(((treatment_metric - treatment_mean) ** 2).mean())

            if np.float(treatment_mean) != 0:
                treatment_LCI = treatment_mean - 1.96 * treatment_std / np.sqrt(treatment_n)
                treatment_UCI = treatment_mean + 1.96 * treatment_std / np.sqrt(treatment_n)
            else:
                treatment_LCI = 0.00001
                treatment_UCI = 0.00001

            ttest_p_value = statsmodels.stats.weightstats.ttest_ind(
                x1=self.weighted_data.loc[self.weighted_data[self.trt] == 1, metric],
                x2=self.weighted_data.loc[self.weighted_data[self.trt] == 0, metric],
                alternative='two-sided',
                weights=(self.weighted_data.loc[self.weighted_data[self.trt] == 1, 'wt'],
                         self.weighted_data.loc[self.weighted_data[self.trt] == 0, 'wt']))[1]

            metric_output.loc[i] = [
                self.trt
                , int(round(self.mt.iloc[:, 0].sum() * self.match_rate_out()[1]))
                , self.match_rate_out()[1]
                , Control
                , int(round(self.mt.iloc[:, 1].sum() * self.match_rate_out()[2]))
                , self.match_rate_out()[2]
                , metric
                , effect[0]
                , treatment_LCI
                , treatment_UCI
                , effect[1]
                , control_LCI
                , control_UCI
                , effect[0] - effect[1]
                , min(treatment_LCI - control_LCI, treatment_LCI - control_UCI, treatment_UCI - control_LCI,
                      treatment_UCI - control_UCI)
                , max(treatment_UCI - control_LCI, treatment_LCI - control_UCI, treatment_UCI - control_LCI,
                      treatment_UCI - control_UCI)
                , (effect[0] - effect[1]) / effect[1]
                , min((treatment_LCI - control_LCI) / control_LCI, (treatment_LCI - control_UCI) / control_UCI,
                      (treatment_UCI - control_LCI) / control_LCI, (treatment_UCI - control_UCI) / control_UCI)
                , max(
                    (treatment_LCI - control_LCI) / control_LCI, (treatment_LCI - control_UCI) / control_UCI,
                    (treatment_UCI - control_LCI) / control_LCI, (treatment_UCI - control_UCI) / control_UCI

                )
                , ttest_p_value
            ]
            i = + i + 1

        return metric_output
