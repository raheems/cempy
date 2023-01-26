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

    def __init__(self, data, treatment, metrics=[], matchvar=[], exclude=[]):
        self.matchvar = [i for i in matchvar if i not in exclude + metrics]

        # Data cleaning process
        # Remove missing values if found in matchvar

        self.data = data.dropna(subset=matchvar)
        self.treatment = treatment
        self.metrics = metrics

        # Denifing the bin signatures
        self.data['bin_sig'] = self.data[self.matchvar[0]].astype(str)
        for i in range(1, len(self.matchvar)):
            self.data['bin_sig'] = self.data['bin_sig'].astype(str) + '_' + self.data[self.matchvar[i]].astype(str)

        # Define treatment control dataframes
        self.t = self.data.loc[self.data[treatment] == 1]
        self.c = self.data.loc[self.data[treatment] == 0]

        # Other attributes

        # CEM summary table
        self.ms = self.cem_summary(self.treatment)
        self.nt_matched = self.ms.groupby(['is_matched']).sum().iloc[1, 0]
        self.nc_matched = self.ms.groupby(['is_matched']).sum().iloc[1, 1]

        # CEM summary with weights for each record
        self.msw = self.get_weights()
        self.msw[self.treatment] = 0
        self.mt = self.matched_table()

        # Data with weights
        self.weighted_data = self.get_weighted_data()


    # Return the list of matched variables
    def matched_var(self):
        return list(self.matchvar)


    # Create CEM summary

    def cem_summary(self, treatment=False):
        """
        cem_summary method performes many-to-many matching and returns a copy of the original `dataframe` that
        contains addition column called 'is_matched' which indicates whether a member has a match or not
        """

        # Group members by bin signature andc ount how many members belong to each type

        self.t_grouped = self.t.groupby(['bin_sig']).size().to_frame(name='trt_count').reset_index()
        self.c_grouped = self.c.groupby(['bin_sig']).size().to_frame(name='ctl_count').reset_index()

        # Combine all grouped members into a single table
        self.member_summary = pd.merge(self.t_grouped, self.c_grouped, how='outer', on=['bin_sig'])

        # Replace NAs with 0
        self.member_summary = self.member_summary.fillna(0)

        # Add indicator column for matched bin signature
        self.member_summary['is_matched'] = 0
        self.member_summary.loc[
            (self.member_summary['trt_count'] > 0) & (self.member_summary['ctl_count'] > 0), 'is_matched'] = 1

        return self.member_summary


    def get_matched(self):
        # 1. Matching
        self.matched_data = self.cem_summary(self.treatment)

        # Get a list of bins that are matched
        bin_matched = self.matched_data.loc[self.matched_data['is_matched'] == 1]['bin_sig']
        bin_matched = list(bin_matched)

        # Get the matched members out
        data_matched = self.data[self.data['bin_sig'].isin(bin_matched)]

        return data_matched


    def get_unmatched(self):
        # 1. Matching
        self.matched_data = self.cem_summary(self.treatment)

        # Get a list of bins that are matched
        bin_unmatched = self.matched_data.loc[self.matched_data['is_matched'] == 0]['bin_sig']
        bin_unmatched = list(bin_unmatched)

        # Get matched members out
        data_unmatched = self.data[self.data['bin_sig'].isin(bin_unmatched)]

        return data_unmatched


    def matched_table(self):
        """
        Returns a table of matched or unmatched members in the treatment and control groups
        """

        summary_matched = self.cem_summary(self.treatment)[['trt_count', 'ctl_count', 'is_matched']]

        return summary_matched.groupby(['is_matched']).sum()


    def match_rate_out(self):
        # Calculate matching rates: overall, treatment and control

        overall = (self.mt.iloc[1].sum()) / (self.mt.sum().sum())
        treatment = (self.mt.iloc[1, 0].sum()) / (self.mt.iloc[:, 0].sum())
        control = (self.mt.iloc[1, 1].sum()) / (self.mt.iloc[:, 1].sum())

        return (overall, treatment, control)


    def get_weights(self):
        """
        CEM summary with weight column added
        """

        # Matched control members get normalized weights according to treatment group
        members_summary_matched = self.ms.loc[(self.ms['is_matched'] > 0)]
        members_summary_matched['wt'] = (members_summary_matched['trt_count'] / members_summary_matched['ctl_count']) * \
                                        (self.nc_matched / self.nt_matched)

        return members_summary_matched


    def get_weighted_data(self):
        dfweighted = self.get_weights()
        data_matched = self.get_matched()

        todrop = ['trt_count', 'ctl_count', 'is_matched']
        dfweighted = dfweighted.drop(columns=todrop)

        # Match corresponding weight for every member
        data_matched['bin_sig'] = data_matched['bin_sig'].astype(str)
        dfweighted['bin_sig'] = dfweighted['bin_sig'].astype(str)
        data_matched_w_weights = pd.merge(data_matched, dfweighted, how='left', on=['bin_sig'])

        # Treatment cases get a weight of 1
        data_matched_w_weights.loc[data_matched_w_weights[self.treatment] == 1, 'wt'] = 1
        data_matched_w_weights = data_matched_w_weights.reset_index(drop=True)

        return data_matched_w_weights


    def _effect_out(self, metric):
        """
        Internal function for use within the metric_summary()
        """

        temp = self.data[[self.treatment, 'bin_sig', metric]]
        temp = temp.groupby(['bin_sig', self.treatment]).sum().reset_index()
        effect = pd.merge(temp, self.msw, how='left', on=['bin_sig'])
        effect = effect.fillna(0)

        # Treatment effect
        treatment_effect = ((effect[self.treatment + '_x'] * effect['is_matched'] * effect[metric]).sum()) / self.nt_matched
        control_effect = ((1 - effect[self.treatment + '_x']) * effect['is_matched'] * effect[metric] * \
                          effect['wt']).sum() / self.nc_matched

        return (treatment_effect, control_effect)


    def metric_summary(self, metrics=False):
        if not metrics:
            metrics = self.metrics

        # Remove constant 0 columns from the list of metrics to run
        # data_matched_weights = self.get_weighted_data()
        # mt = self.matched_table()

        Control = 'Control'

        i = 1
        results = pd.DataFrame(
            columns=[
                'Treatment'
                , 'Matched Treatment group - size'
                , 'Treatment group - match rate'
                , 'Control'
                , 'Matched Control group - size'
                , 'Control group - match rate'
                , 'Metric'
                , 'Metric (Treatment)'
                , 'Metric (Treatment) LCF'
                , 'Metric (Treatment) UCF'
                , 'Metric (Control)'
                , 'Metric (Control) LCF'
                , 'Metric (Control) UCF'
                , 'Impact'
                , 'Impact (LCF)'
                , 'Impact (UCF)'
                , 'Impact (%)'
                , 'Impact (%) LCF'
                , 'Impact (%) UCF'
                , 'p-value'
            ]
        )

        control_weights = self.weighted_data.loc[self.weighted_data[self.treatment] == 0, 'wt']
        control_n = (self.weighted_data[self.treatment] == 0).sum()
        treatment_n = (self.weighted_data[self.treatment] == 1).sum()

        for metric in metrics:

            effect = self._effect_out(metric)

            control_metric = self.weighted_data.loc[self.weighted_data[self.treatment] == 0, metric]
            control_mean = np.average(control_metric, weights=control_weights)
            control_std = np.sqrt(np.average((control_metric - control_mean) ** 2, weights=control_weights))

            if np.float(control_mean) != 0:
                control_LCI = control_mean - 1.96 * control_std / np.sqrt(control_n)
                control_UCI = control_mean + 1.96 * control_std / np.sqrt(control_n)
            else:
                control_LCI = 0.00001
                control_UCI = 0.00001

            treatment_metric = self.weighted_data.loc[self.weighted_data[self.treatment] == 1, metric]
            treatment_mean = treatment_metric.mean()
            treatment_std = np.sqrt(((treatment_metric - treatment_mean) ** 2).mean())

            if np.float(treatment_mean) != 0:
                treatment_LCI = treatment_mean - 1.96 * treatment_std / np.sqrt(treatment_n)
                treatment_UCI = treatment_mean + 1.96 * treatment_std / np.sqrt(treatment_n)
            else:
                treatment_LCI = 0.00001
                treatment_UCI = 0.00001

            ttest_p_value = statsmodels.stats.weightstats.ttest_ind(
                x1=self.weighted_data.loc[self.weighted_data[self.treatment] == 1, metric],
                x2=self.weighted_data.loc[self.weighted_data[self.treatment] == 0, metric],
                alternative='two-sided',
                weights=(self.weighted_data.loc[self.weighted_data[self.treatment] == 1, 'wt'],
                         self.weighted_data.loc[self.weighted_data[self.treatment] == 0, 'wt']))[1]

            results.loc[i] = [
                self.treatment
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

        return results