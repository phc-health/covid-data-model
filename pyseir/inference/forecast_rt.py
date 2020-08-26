import math, random
import us
from datetime import datetime, timedelta
import numpy as np
import logging
import pandas as pd
import os, sys, glob
from matplotlib import pyplot as plt
import us
import structlog

# from pyseir.utils import AggregationLevel, TimeseriesType
from pyseir.utils import get_run_artifact_path, RunArtifact
from structlog.threadlocal import bind_threadlocal, clear_threadlocal, merge_threadlocal
from structlog import configure
from enum import Enum

from tensorflow import keras
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

# Hyperparameter Search
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

# Custom Loss Function
import keras.backend as K

# Compute lag
from scipy import signal

# 2d bins statistics
from scipy.stats import binned_statistic

# Feature processing
from pandas.core.window.indexers import (
    BaseIndexer,
    FixedForwardWindowIndexer,
    VariableOffsetWindowIndexer,
)

# Linear Regression
from sklearn import linear_model

# Save MinMaxScaler Dictionary
import pickle

# Aesthetics
from cycler import cycler
import seaborn as sns
from pandas.plotting import scatter_matrix

# Reduce LR
from keras.callbacks import ReduceLROnPlateau

configure(processors=[merge_threadlocal, structlog.processors.KeyValueRenderer()])
log = structlog.get_logger(__name__)


class ForecastRt:
    """
    Write doc string
    """

    def __init__(self, df_all=None):
        self.train = False
        self.infer_only_n_days = 7
        self.scaling_dictionary_file = (
            "../covid-data-public/forecast_data/models/scaling_dictionary.pkl"
        )
        self.model_file = "../covid-data-public/forecast_data/models/model.h5"
        self.save_csv_output = False  # do not set to true for github actions run
        self.csv_output_folder = "./csv_files/"
        self.df_all = df_all
        self.states = "All"  # All to use All
        self.csv_path = "../covid-data-public/forecast_data/merged_delphi_df.csv"
        self.csv_test_path = "../covid-data-public/forecast_data/merged_delphi_df_latest.csv"

        self.merged_df = True  # set to true if input dataframe merges all areas
        self.states_only = True  # set to true if you only want to train on state level data (county level training not implemented...yet)
        self.ref_date = datetime(year=2020, month=1, day=1)
        self.debug_plots = False
        self.debug_output = False
        # Variable Names
        self.aggregate_level_name = "aggregate_level"
        self.state_aggregate_level_name = "state"
        self.state_var_name = "state"
        self.fips_var_name = "fips"  # name of fips var in input csv
        self.fips_var_name_int = (
            "fips_int"  # name of fips used in forecast (cast from input string to int)
        )
        self.quick_test = False
        self.sim_date_name = "sim_day"
        self.index_col_name_csv = "date"
        self.cases_cumulative = True
        self.deaths_cumulative = True
        self.case_var = "cases"
        self.death_var = "deaths"
        self.daily_var_prefix = "new_"
        self.daily_case_var = self.daily_var_prefix + self.case_var
        self.daily_death_var = self.daily_var_prefix + self.death_var
        # self.predict_variable = "Rt_MAP__new_cases"

        self.raw_predict_variable = self.daily_case_var
        self.predict_variable = "smooth_future_new_cases"
        self.d_predict_variable = f"d_{self.predict_variable}"
        self.predict_var_input_feature = (
            False  # set to true to include predict variable in input data
        )
        self.window_size = 7
        self.smooth_variables = [
            self.daily_case_var,
            self.daily_death_var,
            "new_positive_tests",  # calculated by diffing input 'positive_tests' column
            "new_negative_tests",  # calculated by diffing input 'negative_tests' column
            "raw_search",  # raw google health trends data
            "raw_cli",  # fb raw covid like illness
            "raw_ili",  # fb raw flu like illness
            "contact_tracers_count",
            "raw_community",
            "raw_hh_cmnty_cli",
            "raw_nohh_cmnty_cli",
            "raw_wcli",
            "raw_wili",
            "unsmoothed_community",
            "median_home_dwell_time_prop",
            "full_time_work_prop",
            "part_time_work_prop",
            "completely_home_prop",
        ]
        self.forecast_variables = [
            self.predict_variable,
            f"smooth_{self.daily_case_var}",
            "Rt_MAP__new_cases",
            f"smooth_{self.daily_death_var}",
            "smooth_new_negative_tests",  # calculated by diffing input 'negative_tests' column
            "smooth_median_home_dwell_time_prop",
            "smooth_full_time_work_prop",
            "smooth_part_time_work_prop",
            "smooth_completely_home_prop",
            "d_smooth_median_home_dwell_time_prop",
            "d_smooth_full_time_work_prop",
            "d_smooth_part_time_work_prop",
            "d_smooth_completely_home_prop",
            self.fips_var_name_int,
            "smooth_contact_tracers_count",  # number of contacts traced
            "smoothed_cli",  # estimated percentage of covid doctor visits
            "smoothed_hh_cmnty_cli",
            "smoothed_nohh_cmnty_cli",
            "smoothed_ili",
            "smoothed_wcli",
            "smoothed_wili",
            "smoothed_search",  # smoothed google health trends da
            "nmf_day_doc_fbc_fbs_ght",  # delphi combined indicator
            "nmf_day_doc_fbs_ght",
            # Not using variables below
            # "smooth_raw_cli",  # fb raw covid like illness
            # "smooth_raw_ili",  # fb raw flu like illness
            # "smooth_raw_search",  # raw google health trends data
            # "smooth_raw_community",
            # "smooth_raw_hh_cmnty_cli", #fb estimated cli count including household
            # "smooth_raw_nohh_cmnty_cli", #fb estimated cli county not including household
            # "smooth_raw_wcli", #fb cli adjusted with weight surveys
            # "smooth_raw_wili", #fb ili adjusted with weight surveys
            # "smooth_unsmoothed_community",
            # "smoothed_community",
        ]
        self.scaled_variable_suffix = "_scaled"

        # Seq2Seq Parameters
        self.max_scaling = 2  # multiply max feature values by this number for scaling set
        self.min_scaling = 0.5  # multiply min feature values by this number of scaling set
        self.days_between_samples = 1
        self.mask_value = -10
        self.min_number_of_days = 31
        self.sequence_length = (
            30  # can pad sequence with numbers up to this length if input lenght is variable
        )
        self.sample_train_length = 30  # Set to -1 to use all historical data
        self.predict_days = 1
        self.percent_train = False
        self.train_size = 0.8
        self.n_test_days = 10
        self.n_batch = 50
        self.n_epochs = 1000
        self.n_hidden_layer_dimensions = 100
        self.dropout = 0
        self.patience = 30
        self.validation_split = 0  # currently using test set as validation set
        self.hyperparam_search = False
        self.use_log_predict_var = True

    @classmethod
    def run_forecast(cls, df_all=None):
        engine = cls(df_all)
        return engine.forecast()
        # return engine.infer()

    def get_forecast_dfs(self, csvfile, mode):
        if self.merged_df is None or not self.states_only:
            raise NotImplementedError("Only states are supported.")

        df_merge = pd.read_csv(
            csvfile,
            parse_dates=True,
            index_col=self.index_col_name_csv,
            converters={self.fips_var_name: str},
        )

        if self.save_csv_output:
            df_merge.to_csv(self.csv_output_folder + "MERGED_CSV.csv")

        # only store state information
        df_states_merge = df_merge[
            df_merge[self.aggregate_level_name] == self.state_aggregate_level_name
        ]
        # create separate dataframe for each state
        state_df_dictionary = dict(iter(df_states_merge.groupby(self.fips_var_name)))

        # process dataframe
        state_names, df_forecast_list, df_list, df_invalid_list = [], [], [], []
        for state in state_df_dictionary:
            df = state_df_dictionary[state]
            state_name = df[self.fips_var_name][0]
            if self.quick_test and int(state_name) > 3:  # only test two states if quick_test
                continue
            if self.deaths_cumulative:
                df[self.daily_case_var] = df[self.case_var].diff()
            if self.cases_cumulative:
                df[self.daily_death_var] = df[self.death_var].diff()
            df["new_positive_tests"] = df["positive_tests"].diff()
            df["new_negative_tests"] = df["negative_tests"].diff()

            for var in self.smooth_variables:
                df[f"smooth_{var}"] = df.iloc[:][var].rolling(window=self.window_size).mean()
                df[f"d_smooth_{var}"] = df[f"smooth_{var}"].diff()
            # Calculate average of predict variable
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=self.window_size)
            df[self.predict_variable] = (
                df.iloc[:][self.raw_predict_variable].rolling(window=indexer).mean()
            )  # this just grabs the value of the variable 5 days forward, it is not a mean and I dunno why
            if self.use_log_predict_var:
                df[self.predict_variable] = np.log10(df[self.predict_variable])
                df[self.predict_variable] = df[self.predict_variable].replace([np.inf, -np.inf], 0)
                df[self.predict_variable] = df[self.predict_variable].fillna(self.mask_value)
            # Calculate Rt derivative, exclude first row since-- zero derivative
            df[self.d_predict_variable] = df[self.predict_variable].diff()

            # Only keep data points where predict variable exists
            first_valid_index = df[self.predict_variable].first_valid_index()
            last_valid_index = df[self.predict_variable].last_valid_index()
            df_invalid = df[:-1].copy()  # because last entry is from raw data is NaN TBD
            if self.quick_test:
                self.n_batch = 1
                df = df[first_valid_index:].copy()
                df = df[:60].copy()
            if mode == "infer":
                start_date = self.infer_only_n_days + self.min_number_of_days + 19
                end_date = start_date - 1
                df = df[-(self.min_number_of_days + 16 + 3) : -16].copy()
                print(df)

            df = df[first_valid_index:last_valid_index].copy()

            # dates.append(df.iloc[-self.predict_days:]['sim_day'])
            # TODO decide if first and last entry need to be removed

            df = df[1:]
            df[self.fips_var_name_int] = df[self.fips_var_name].astype(int)
            df[self.sim_date_name] = (df.index - self.ref_date).days + 1
            df_forecast = df.fillna(self.mask_value)

            df_invalid[self.fips_var_name_int] = df_invalid[self.fips_var_name].astype(int)
            df_invalid[self.sim_date_name] = (df_invalid.index - self.ref_date).days + 1
            df_forecast_invalid = df_invalid.fillna(self.mask_value)

            state_names.append(state_name)
            df_forecast_list.append(df_forecast)
            df_invalid_list.append(df_forecast_invalid)
            df_list.append(df)
            df_slim = slim(df_forecast, self.forecast_variables)

            if 1 == 0:
                # if self.debug_plots:
                corr = df_slim.corr()
                plt.close("all")
                ax = sns.heatmap(
                    corr,
                    vmin=-1,
                    vmax=1,
                    center=0,
                    cmap=sns.diverging_palette(20, 220, n=200),
                    square=True,
                    annot=True,
                    annot_kws={"size": 4},
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
                plt.savefig(
                    self.csv_output_folder + us.states.lookup(state_name).name + "_corr.pdf",
                    bbox_inches="tight",
                )
                plt.close("all")
                axs = pd.plotting.scatter_matrix(df_slim)
                n = len(df_slim.columns)
                for x in range(n):
                    for y in range(n):
                        # to get the axis of subplots
                        ax = axs[x, y]
                        # to make x axis name vertical
                        ax.xaxis.label.set_rotation(90)
                        # to make y axis name horizontal
                        ax.yaxis.label.set_rotation(0)
                        # to make sure y axis names are outside the plot area
                        ax.yaxis.labelpad = 50

                plt.savefig(
                    self.csv_output_folder + us.states.lookup(state_name).name + "_scatter.pdf",
                    bbox_inches="tight",
                )
            if self.save_csv_output:
                df_forecast.to_csv(self.csv_output_folder + df["state"][0] + "_forecast.csv")
                df.to_csv(self.csv_output_folder + df["state"][0] + "_OG_forecast.csv")

        return state_names, df_forecast_list, df_invalid_list

    def get_train_test_samples(self, df_forecast):
        # True if test samples are constrained to be latest samples
        test_set_end_of_sample = False

        # create list of dataframe samples
        df_samples = self.create_samples(df_forecast)

        # Split sample list into training and testing
        if test_set_end_of_sample:
            # Determine size of train set to split sample list into training and testing
            if self.percent_train:
                train_set_length = int(len(df_samples) * self.train_size)
            else:
                train_set_length = int(len(df_samples)) - self.n_test_days
            train_samples_not_spaced = df_samples[:train_set_length]
            first_test_index = (
                self.days_between_samples * ((train_set_length // self.days_between_samples) + 1)
                - 1
            )
            test_samples = df_samples[first_test_index:]
            if 1 == 0:
                # if self.save_csv_output:
                for i in range(len(train_samples_not_spaced)):
                    df = train_samples_not_spaced[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_train-notspaced.csv")

                for i in range(len(test_samples)):
                    df = test_samples[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_test-notspaced.csv")
            # For traning only keep samples that are days_between_samples apart (avoid forecast learning meaningless correlations between labels)
            train_samples = train_samples_not_spaced[0 :: self.days_between_samples]

        else:  # test and train set randomly selected from sample set
            # require samples to be days_between_samples apart
            df_samples_spaced = df_samples[0 :: self.days_between_samples]

            if self.percent_train:
                train_set_length = int(len(df_samples_spaced) * self.train_size)
            else:
                train_set_length = int(len(df_samples_spaced)) - self.n_test_days
            # shuffle samples before spliting between test and train sets
            random.shuffle(df_samples_spaced)
            # Select train and test sets
            train_samples = df_samples_spaced[:train_set_length]
            test_samples = df_samples_spaced[train_set_length:]

            if 1 == 0:
                # if self.save_csv_output:
                for i in range(len(train_samples_not_spaced)):
                    df = train_samples_not_spaced[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_train-notspaced.csv")

                for i in range(len(test_samples)):
                    df = test_samples[i]
                    if self.save_csv_output:
                        df.to_csv(self.csv_output_folder + "df" + str(i) + "_test-notspaced.csv")

        # Scaling set is the concatenated train_samples
        scaling_set = pd.concat(train_samples)

        return train_samples, test_samples, scaling_set

    def plot_variables(self, df_list, state_fips, scalers_dict):
        col = plt.cm.jet(np.linspace(0, 1, round(len(self.forecast_variables) + 1)))
        BOLD_LINEWIDTH = 3
        for df, state in zip(df_list, state_fips):
            if self.save_csv_output:
                df.to_csv(self.csv_output_folder + us.states.lookup(state).name + "forecast.csv")
            fig, ax = plt.subplots(figsize=(18, 12))
            for var, color in zip(self.forecast_variables, col):
                if var != self.predict_variable:
                    ax.plot(df.index, df[var], label=var, color=color, linewidth=1)
                else:
                    ax.plot(df[var], label=var, color="black", linewidth=BOLD_LINEWIDTH)
            ax.legend()
            plt.xticks(rotation=30, fontsize=14)
            plt.grid(which="both", alpha=0.5)
            output_path = get_run_artifact_path(state, RunArtifact.FORECAST_VAR_UNSCALED)
            plt.title(us.states.lookup(state).name)
            # plt.savefig(output_path, bbox_inches="tight")

            fig2, ax2 = plt.subplots(figsize=(18, 12))
            # for var, color in zip(self.forecast_variables, col):
            scaled_variables = {}
            for i in range(len(self.forecast_variables)):
                if i % 2 == 0:
                    lstyle = "solid"
                elif i % 3 == 0:
                    lstyle = "dotted"
                else:
                    lstyle = "dashed"
                var = self.forecast_variables[i]
                color = col[i]
                reshaped_data = df[var].values.reshape(-1, 1)
                scaled_values = scalers_dict[var].transform(reshaped_data)
                scaled_variables[var] = scaled_values
                # ax2.plot(scaled_values, label=var, color=color)
                if var != self.predict_variable and var != self.daily_case_var:
                    ax2.plot(
                        df.index,
                        scaled_values,
                        label=var,
                        color=color,
                        linewidth=1,
                        linestyle=lstyle,
                    )
                else:
                    ax2.plot(
                        df.index,
                        scaled_values,
                        label=var,
                        color=color,
                        linewidth=BOLD_LINEWIDTH,
                        linestyle=lstyle,
                    )
            ax2.legend()
            plt.xticks(rotation=30, fontsize=14)
            plt.grid(which="both", alpha=0.5)
            plt.title(us.states.lookup(state).name)
            plt.ylim(-1, 2)
            output_path = get_run_artifact_path(state, RunArtifact.FORECAST_VAR_SCALED)
            plt.savefig(output_path, bbox_inches="tight")

            plt.close("all")

            # log.info('scaled values dictionary')
            # log.info(scaled_variables)

            """
            for var, values in scaled_variables.items():
              log.info(var)
              log.info(values)
              log.info(self.predict_variable)
              log.info(scaled_values[self.predict_variable])
              lag = align_time_series(values, scaled_values[self.predict_variable])
              plt.close('all')
              plt.plot(values, label = var)
              plt.plot(scaled_values, label = self.predict_variable)
              plt.legend()
              plt.title(us.states.lookup(state).name)
              plt.xticks(rotation=30, fontsize=14)
              plt.savefig(self.csv_output_folder + us.states.lookup(state).name+ var + '.pdf')
              log.info(f'{var} lag: {lag}')
            """

        return

    def infer(self):
        compute_metrics = True
        area_fips, area_df_list, area_df_invalid_list = self.get_forecast_dfs(
            self.csv_path, "infer"
        )
        # Load Scaling Dictionary
        scalers_file = open(self.scaling_dictionary_file, "rb")
        scalers_dict = pickle.load(scalers_file)
        # Load Model
        model = keras.models.load_model(self.model_file, custom_objects={"smape": smape})

        # Arrays for storing prediction metrics
        actuals_test = []
        test_linear_mape = []
        test_linear_mae = []
        test_linear_smape = []
        test_forecast_mae = []
        test_forecast_mape = []
        test_forecast_smape = []
        area_test_samples = []

        for df, fips in zip(area_df_list, area_fips):
            samples = self.create_samples(df)
            slimmed_samples = slim(samples, self.forecast_variables)
            test_X, test_Y = self.get_scaled_X_Y(slimmed_samples, scalers_dict, "future")
            (
                forecasts_test,
                dates_test,
                unscaled_forecasts_test,
                regression_predictions_test,
            ) = self.get_forecasts(samples, test_X, test_Y, scalers_dict, model)
            plt.figure(figsize=(18, 12))
            fips = samples[0]["fips"][0]
            state_name = us.states.lookup(fips).name
            plt.plot(
                np.squeeze(dates_test),
                np.squeeze(forecasts_test),
                color="orange",
                label="Test",
                linewidth=0,
                markersize=5,
                marker="*",
            )
            plt.scatter(
                np.squeeze(dates_test),
                np.squeeze(regression_predictions_test),
                color="purple",
                label="Test Linear",
                marker="o",
            )
            plt.plot(
                df.index,
                df[self.predict_variable],
                label=self.predict_variable,
                markersize=3,
                marker=".",
                linewidth=3,
            )
            plt.xlabel(self.sim_date_name)
            plt.ylabel(self.predict_variable)
            plt.legend()
            plt.grid(which="both", alpha=0.5)
            plt.title(state_name)

            # Compute metrics
            if compute_metrics:
                test_labels = scalers_dict[self.predict_variable].inverse_transform(
                    test_Y.reshape(1, -1)
                )

                # linear regression preformance metrics
                mae_test_linear, mape_test_linear, smape_test_linear = get_error_metrics(
                    test_labels, regression_predictions_test
                )

                # forecast preformance metrics
                mae_test_forecast, mape_test_forecast, smape_test_forecast = get_error_metrics(
                    test_labels, forecasts_test
                )
                seq_params_dict = {
                    "MAE: test forecast": np.mean(mae_test_forecast),
                    "MAE: test linear": np.mean(mae_test_linear),
                    "MAPE: test forecast": np.mean(mae_test_forecast),
                    "MAPE: test linear": np.mean(mae_test_linear),
                    "SMAPE: test forecast": np.mean(smape_test_forecast),
                    "SMAPE: test linear": np.mean(smape_test_linear),
                }
                for i, (k, v) in enumerate(seq_params_dict.items()):
                    plt.text(
                        1.0,
                        0.7 - 0.032 * i,
                        f"{k}={v:1.1f}",
                        transform=plt.gca().transAxes,
                        fontsize=15,
                        alpha=0.6,
                    )

                # Append metrics to arrays for meta metrics analysis
                actuals_test.extend(np.squeeze(test_labels))
                # Linear Predictions
                test_linear_mae.extend(mae_test_linear)
                test_linear_mape.extend(mape_test_linear)
                test_linear_smape.extend(smape_test_linear)
                # Forecast Predictions
                test_forecast_mae.extend(mae_test_forecast)
                test_forecast_mape.extend(mape_test_forecast)
                test_forecast_smape.extend(smape_test_forecast)

            output_path = get_run_artifact_path(fips, RunArtifact.FORECAST_RESULT)
            state_obj = us.states.lookup(state_name)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")

        plot_percentile_error(
            None, actuals_test, None, test_forecast_mae, "mae", self.predict_variable,
        )
        plot_percentile_error(
            None, actuals_test, None, test_forecast_mape, "mape", self.predict_variable,
        )
        plot_percentile_error(
            None, actuals_test, None, test_forecast_smape, "smape", self.predict_variable,
        )

        if self.debug_plots:
            self.plot_variables(
                slim(area_df_list, self.forecast_variables), area_fips, scalers_dict
            )

    def forecast(self):
        """
        predict r_t for 14 days into the future
        Parameters
        df_all: dataframe with dates, new_cases, other specified features
        Potential todo: add more features #ALWAYS
        Returns
        dates and forecast r_t values
        """

        # split merged dataframe into state level dataframes (this includes adding variables and masking nan values)
        area_fips, area_df_list, area_df_invalid_list = self.get_forecast_dfs(
            self.csv_path, "train"
        )
        # get train, test, and scaling samples per state and append to list
        area_scaling_samples, area_train_samples, area_test_samples = [], [], []
        for df, fips in zip(area_df_list, area_fips):
            train, test, scaling = self.get_train_test_samples(df)
            area_scaling_samples.append(scaling)
            area_train_samples.append(train)
            area_test_samples.append(test)
            area_name = us.states.lookup(fips).name
            log.info(f"{area_name}: train_samples: {len(train)} test_samples: {len(test)}")
        # Get scaling dictionary
        # TODO add max min rows to avoid domain adaption issues
        train_scaling_set = pd.concat(area_scaling_samples)
        print(train_scaling_set.max())
        scalers_dict = self.get_scaling_dictionary(slim(train_scaling_set, self.forecast_variables))
        output_path = get_run_artifact_path(fips, RunArtifact.SCALING_DICTIONARY)
        f = open(output_path, "wb")
        pickle.dump(scalers_dict, f)
        f.close()

        if self.debug_plots:
            self.plot_variables(
                slim(area_df_invalid_list, self.forecast_variables), area_fips, scalers_dict
            )

        # Create scaled train samples
        list_train_X, list_train_Y, list_test_X, list_test_Y = [], [], [], []
        # iterate over train/test_samples = list[state_dfs_samples]
        log.info("slimming samples")  # THIS IS THE SLOW STEP
        for train, test in zip(area_train_samples, area_test_samples):
            train_filter = slim(train, self.forecast_variables)
            test_filter = slim(test, self.forecast_variables)
            train_X, train_Y = self.get_scaled_X_Y(train_filter, scalers_dict, "train")
            test_X, test_Y = self.get_scaled_X_Y(test_filter, scalers_dict, "test")
            list_train_X.append(train_X)
            list_train_Y.append(train_Y)
            list_test_X.append(test_X)
            list_test_Y.append(test_Y)
        log.info("samples slimmed")
        final_list_train_X = np.concatenate(list_train_X)
        final_list_train_Y = np.concatenate(list_train_Y)
        final_list_test_X = np.concatenate(list_test_X)
        final_list_test_Y = np.concatenate(list_test_Y)

        skip_train = 37  # 47
        skip_test = 20  # 10
        if self.quick_test:
            skip_train = 0
            skip_test = 0
        if skip_train > 0:
            final_list_train_X = final_list_train_X[:-skip_train]
            final_list_train_Y = final_list_train_Y[:-skip_train]
        if skip_test > 0:
            final_list_test_X = final_list_test_X[:-skip_test]
            final_list_test_Y = final_list_test_Y[:-skip_test]

        log.info(f"train: {len(final_list_train_X)} test: {len(final_list_test_X)}")
        if self.hyperparam_search:
            model, history, tuner = self.build_model(
                final_list_train_X, final_list_train_Y, final_list_test_X, final_list_test_Y,
            )
            best_hps = tuner.get_best_hyperparameters()[0]
            dropout = best_hps.get("dropout")
            n_hidden_layer_dimensions = best_hps.get("n_hidden_layer_dimensions")
            n_layers = best_hps.get("n_layers")
        dropout = 0
        n_hidden_layer_dimensions = 100
        n_layers = 4
        log.info(f"n_features: {len(self.forecast_variables)}")
        if self.predict_var_input_feature:
            n_features = len(self.forecast_variables)
        else:
            n_features = len(self.forecast_variables) - 1
        modelClass = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=n_features,
            mask_value=self.mask_value,
            batch_size=self.n_batch,
        )
        model = modelClass.build(
            tune=False,
            dropout=dropout,
            n_hidden_layer_dimensions=n_hidden_layer_dimensions,
            n_layers=n_layers,
        )
        es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=self.patience)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="fit_logs")
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)
        history = model.fit(
            final_list_train_X,
            final_list_train_Y,
            epochs=self.n_epochs,
            batch_size=self.n_batch,
            callbacks=[es, tensorboard_callback],
            verbose=1,
            shuffle=True,
            validation_data=(final_list_test_X, final_list_test_Y),
        )

        model.evaluate(final_list_train_X, final_list_train_Y)  # this gives actual loss

        forecast_model_skeleton = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=n_features,
            mask_value=self.mask_value,
            batch_size=1,
        )
        forecast_model = forecast_model_skeleton.build(
            hp=None,
            tune=False,
            n_layers=n_layers,
            dropout=dropout,
            n_hidden_layer_dimensions=n_hidden_layer_dimensions,
        )
        trained_model_weights = model.get_weights()
        forecast_model.set_weights(trained_model_weights)
        # Save trained model that has batch size 1 for future forecasts
        output_path = get_run_artifact_path(fips, RunArtifact.MODEL)
        forecast_model.save(output_path)

        # Arrays for storing prediction metrics
        actuals_train = []
        actuals_test = []

        train_linear_mape = []
        train_linear_mae = []
        train_linear_smape = []
        test_linear_mape = []
        test_linear_mae = []
        test_linear_smape = []

        train_forecast_mae = []
        train_forecast_mape = []
        train_forecast_smape = []
        test_forecast_mae = []
        test_forecast_mape = []
        test_forecast_smape = []

        # Plot Model Predictions
        DATA_LINEWIDTH = 3
        MODEL_LINEWIDTH = 0

        for train_df, train_X, train_Y, test_df, test_X, test_Y, area_df, area_df_invalid in zip(
            area_train_samples,
            list_train_X,
            list_train_Y,
            area_test_samples,
            list_test_X,
            list_test_Y,
            area_df_list,
            area_df_invalid_list,
        ):
            # Test samples not in train/test sets due to incomplete datastreams/no labels
            invalid_dfs = []
            invalid_dfs.append(area_df_invalid[-31:-1])
            invalid_dfs.append(area_df_invalid[-30:])
            invalid_X, invalid_Y = self.get_scaled_X_Y(
                slim(invalid_dfs, self.forecast_variables), scalers_dict, "future"
            )
            (
                forecast_invalid,
                dates_invalid,
                unscaled_forecast_invalid,
                regression_future_invalid,
            ) = self.get_forecasts(
                invalid_dfs, invalid_X, invalid_Y, scalers_dict, forecast_model, "future"
            )

            # Retrieve linear prediction
            (
                forecasts_train,
                dates_train,
                unscaled_forecasts_train,
                regression_predictions_train,
            ) = self.get_forecasts(train_df, train_X, train_Y, scalers_dict, forecast_model)
            print("train forecasts")
            print(forecasts_train)
            print(dates_train)
            (
                forecasts_test,
                dates_test,
                unscaled_forecasts_test,
                regression_predictions_test,
            ) = self.get_forecasts(test_df, test_X, test_Y, scalers_dict, forecast_model)

            plt.figure(figsize=(18, 12))
            fips = train_df[0]["fips"][0]
            state_name = us.states.lookup(fips).name

            if self.predict_days == 1:
                plt.plot(
                    np.squeeze(dates_invalid),
                    np.squeeze(forecast_invalid),
                    color="magenta",
                    label="future",
                    markersize=5,
                    marker="*",
                )
                plt.plot(
                    np.squeeze(dates_train),
                    np.squeeze(forecasts_train),
                    color="green",
                    label="Train",
                    linewidth=MODEL_LINEWIDTH,
                    markersize=5,
                    marker="*",
                )
                plt.plot(
                    np.squeeze(dates_test),
                    np.squeeze(forecasts_test),
                    color="orange",
                    label="Test",
                    linewidth=MODEL_LINEWIDTH,
                    markersize=5,
                    marker="*",
                )
                plt.scatter(
                    np.squeeze(dates_train),
                    np.squeeze(regression_predictions_train),
                    color="purple",
                    label="Train Linear",
                    marker="o",
                )
                plt.scatter(
                    np.squeeze(dates_test),
                    np.squeeze(regression_predictions_test),
                    color="blue",
                    label="Test Linear",
                    marker="o",
                )

                # Retrieve actual data points for preformance metric calculations
                train_labels = scalers_dict[self.predict_variable].inverse_transform(
                    train_Y.reshape(1, -1)
                )
                test_labels = scalers_dict[self.predict_variable].inverse_transform(
                    test_Y.reshape(1, -1)
                )
                if self.use_log_predict_var:
                    test_labels = 10 ** test_labels
                    train_labels = 10 ** train_labels

                # linear regression preformance metrics
                mae_train_linear, mape_train_linear, smape_train_linear = get_error_metrics(
                    train_labels, regression_predictions_train
                )
                mae_test_linear, mape_test_linear, smape_test_linear = get_error_metrics(
                    test_labels, regression_predictions_test
                )

                # forecast preformance metrics
                mae_train_forecast, mape_train_forecast, smape_train_forecast = get_error_metrics(
                    train_labels, forecasts_train
                )
                mae_test_forecast, mape_test_forecast, smape_test_forecast = get_error_metrics(
                    test_labels, forecasts_test
                )

                # Append metrics to arrays for meta metrics analysis
                actuals_train.extend(np.squeeze(train_labels))
                actuals_test.extend(np.squeeze(test_labels))
                # Linear Predictions
                train_linear_mae.extend(mae_train_linear)
                train_linear_mape.extend(mape_train_linear)
                train_linear_smape.extend(smape_train_linear)
                test_linear_mae.extend(mae_test_linear)
                test_linear_mape.extend(mape_test_linear)
                test_linear_smape.extend(smape_test_linear)
                # Forecast Predictions
                train_forecast_mae.extend(mae_train_forecast)
                train_forecast_mape.extend(mape_train_forecast)
                train_forecast_smape.extend(smape_train_forecast)
                test_forecast_mae.extend(mae_test_forecast)
                test_forecast_mape.extend(mape_test_forecast)
                test_forecast_smape.extend(smape_test_forecast)

            else:
                for n in range(len(dates_train)):
                    newdates = dates_train[n]
                    j = np.squeeze(forecasts_train[n])
                    if n == 0:
                        plt.plot(
                            newdates,
                            j,
                            color="green",
                            label="Train Set",
                            linewidth=MODEL_LINEWIDTH,
                            markersize=5,
                            marker="*",
                        )
                    else:
                        plt.plot(
                            newdates, j, color="green", linewidth=MODEL_LINEWIDTH, markersize=0
                        )

                for n in range(len(dates_test)):
                    newdates = dates_test[n]
                    j = np.squeeze(forecasts_test[n])
                    if n == 0:
                        plt.plot(
                            newdates,
                            j,
                            color="orange",
                            label="Test Set",
                            linewidth=MODEL_LINEWIDTH,
                            markersize=5,
                            marker="*",
                        )
                    else:
                        plt.plot(
                            newdates, j, color="orange", linewidth=MODEL_LINEWIDTH, markersize=0
                        )
            if self.use_log_predict_var:
                data = 10 ** area_df[self.predict_variable]
            else:
                data = area_df[self.predict_variable]
            plt.plot(
                area_df.index,
                data,
                label=self.predict_variable,
                markersize=3,
                marker=".",
                linewidth=DATA_LINEWIDTH,
            )
            plt.plot(
                area_df_invalid.index,
                area_df_invalid["smooth_new_cases"],
                label="smooth new cases",
                markersize=3,
                marker=".",
            )

            plt.xlabel(self.sim_date_name)
            plt.ylabel(self.predict_variable)
            plt.legend()
            plt.grid(which="both", alpha=0.5)
            # Seq2Seq Parameters
            seq_params_dict = {
                "days_between_samples": self.days_between_samples,
                "min_number_days": self.min_number_of_days,
                "sequence_length": self.sequence_length,
                "train_length": self.sample_train_length,
                "% train": self.train_size,
                "batch size": self.n_batch,
                "epochs": self.n_epochs,
                "hidden layer dimensions": self.n_hidden_layer_dimensions,
                "dropout": self.dropout,
                "patience": self.patience,
                "validation split": self.validation_split,
                "mask value": self.mask_value,
                "MAE: train forecast": np.mean(mae_train_forecast),
                "MAE: train linear": np.mean(mae_train_linear),
                "MAPE: train forecast": np.mean(mape_train_forecast),
                "MAPE: train linear": np.mean(mape_train_linear),
                "SMAPE: train forecast": np.mean(smape_train_forecast),
                "SMAPE: trian linear": np.mean(smape_train_linear),
                "MAE: test forecast": np.mean(mae_test_forecast),
                "MAE: test linear": np.mean(mae_test_linear),
                "MAPE: test forecast": np.mean(mae_test_forecast),
                "MAPE: test linear": np.mean(mae_test_linear),
                "SMAPE: test forecast": np.mean(smape_test_forecast),
                "SMAPE: test linear": np.mean(smape_test_linear),
            }
            for i, (k, v) in enumerate(seq_params_dict.items()):

                fontweight = "bold" if k in ("important variables") else "normal"

                if np.isscalar(v) and not isinstance(v, str):
                    plt.text(
                        1.0,
                        0.7 - 0.032 * i,
                        f"{k}={v:1.1f}",
                        transform=plt.gca().transAxes,
                        fontsize=15,
                        alpha=0.6,
                        fontweight=fontweight,
                    )

                else:
                    plt.text(
                        1.0,
                        0.7 - 0.032 * i,
                        f"{k}={v}",
                        transform=plt.gca().transAxes,
                        fontsize=15,
                        alpha=0.6,
                        fontweight=fontweight,
                    )

            plt.title(state_name + ": epochs: " + str(self.n_epochs))
            output_path = get_run_artifact_path(fips, RunArtifact.FORECAST_RESULT)
            state_obj = us.states.lookup(state_name)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")

        # Plot loss Plot with Combined Preformance Metrics
        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot(history.history["loss"], color="blue", linestyle="solid", label="Train")
        ax.plot(
            history.history["val_loss"], color="green", linestyle="solid", label="Test",
        )

        plt.legend(loc="upper right")
        plt.title("Loss vs. Epochs")

        textstr = "\n".join(
            (
                "TRAIN (forecast, linear)",
                f"MAE: ({np.mean(train_forecast_mae):.1f}, {np.mean(train_linear_mae):.1f})",
                f"MAPE: ({np.mean(train_forecast_mape):.1f}, {np.mean(train_linear_mape):.1f})",
                f"SMAPE: ({np.mean(train_forecast_smape):.1f}, {np.mean(train_linear_smape):.1f})",
                "",
                "TEST (forecast, linear)",
                f"MAE: ({np.mean(test_forecast_mae):.1f}, {np.mean(test_linear_mae):.1f})",
                f"MAPE: ({np.mean(test_forecast_mape):.1f}, {np.mean(test_linear_mape):.1f})",
                f"SMAPE: ({np.mean(test_forecast_smape):.1f}, {np.mean(test_linear_smape):.1f})",
            )
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=props,
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        output_path = get_run_artifact_path("01", RunArtifact.FORECAST_LOSS)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close("all")

        plot_percentile_error(
            actuals_train,
            actuals_test,
            train_forecast_mae,
            test_forecast_mae,
            "mae",
            self.predict_variable,
        )
        plot_percentile_error(
            actuals_train,
            actuals_test,
            train_forecast_mape,
            test_forecast_mape,
            "mape",
            self.predict_variable,
        )
        plot_percentile_error(
            actuals_train,
            actuals_test,
            train_forecast_smape,
            test_forecast_smape,
            "smape",
            self.predict_variable,
        )

        return

    def get_forecasts(self, df_list, X_list, Y_list, scalers_dict, model, label="none"):
        unscaled_predictions = list()
        forecasts = list()
        dates = list()
        regr_prediction = list()
        actuals = list()
        do_linear = True

        i = 0
        for df, x, y in zip(df_list, X_list, Y_list):
            i += 1
            if do_linear:
                n_days = 11  # number of previous datapoints used in linear interpolation
                predict_out_days = 7  # how many days out to predict

                # Take the last 11 days of input dataframe for linear regression
                df_linear = df.tail(n_days).reset_index(drop=True)
                # train set is everything except the labels for testing
                df_linear_train = df_linear.head(n_days - self.predict_days)
                # labels are the last predict_days rows
                df_linear_test = df.tail(self.predict_days)

                train_X = df_linear_train.index.to_numpy().reshape(-1, 1)
                train_Y = df_linear_train["smooth_new_cases"].to_numpy().reshape(-1, 1)
                actuals.append(int(df_linear_test["smooth_future_new_cases"]))
                regr = linear_model.LinearRegression()
                regr.fit(train_X, train_Y)
                train_prediction_day = train_X[len(train_X) - 1].reshape(1, -1) + predict_out_days
                train_prediction = regr.predict(train_prediction_day)
                if self.use_log_predict_var:
                    train_prediction = 10 ** train_prediction
                regr_prediction.append(train_prediction)

            x = x.reshape(1, x.shape[0], x.shape[1])
            unscaled_prediction = model.predict(x)
            thisforecast = scalers_dict[self.predict_variable].inverse_transform(
                unscaled_prediction
            )
            if self.use_log_predict_var:
                thisforecast = 10 ** thisforecast

            forecasts.append(thisforecast)
            unscaled_predictions.append(unscaled_prediction)
            if label == "none":
                dates.append(df.iloc[-self.predict_days :].index)
            elif label == "future":
                dates.append(df.iloc[-1:].index + timedelta(days=1))

        return forecasts, dates, unscaled_predictions, np.array(regr_prediction)

    def get_scaling_dictionary(self, train_scaling_set):
        scalers_dict = {}
        if self.save_csv_output:
            train_scaling_set.to_csv(self.csv_output_folder + "scalingset_now.csv")
        for columnName, columnData in train_scaling_set.iteritems():
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            reshaped_data = columnData.values.reshape(-1, 1)

            scaler = scaler.fit(reshaped_data)
            # scaled_values = scaler.transform(reshaped_data)

            scalers_dict.update({columnName: scaler})
        return scalers_dict

    def get_scaled_X_Y(self, samples, scalers_dict, label):
        sample_list = list()
        for sample in samples:
            for columnName, columnData in sample.iteritems():
                scaled_values = scalers_dict[columnName].transform(columnData.values.reshape(-1, 1))
                sample.loc[:, f"{columnName}{self.scaled_variable_suffix}"] = scaled_values
            sample_list.append(sample)
        X, Y = self.get_X_Y(sample_list, label)
        return X, Y

    def old_specify_model(
        self, n_batch
    ):  # , sample_train_length, n_features, predict_sequence_length):
        model = Sequential()
        model.add(
            Masking(
                mask_value=self.mask_value,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
                stateful=True,
                return_sequences=True,
            )
        )
        model.add(
            LSTM(
                self.n_hidden_layer_dimensions,
                batch_input_shape=(n_batch, self.sequence_length, len(self.forecast_variables)),
                stateful=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(Dense(self.predict_days))

        return model

    def build_model(self, final_train_X, final_train_Y, final_test_X, final_test_Y):
        hypermodel = MyHyperModel(
            train_sequence_length=self.sequence_length,
            predict_sequence_length=self.predict_days,
            n_features=len(self.forecast_variables),
            mask_value=self.mask_value,
            batch_size=self.n_batch,
        )
        tuner = RandomSearch(
            hypermodel,
            objective="val_loss",
            max_trials=100,
            directory="hyperparam_search",
            project_name="hyperparam_search",
        )

        # final_train_X = final_train_X[:-2]
        # final_train_Y = final_train_Y[:-2]
        tuner.search(
            final_train_X,
            final_train_Y,
            epochs=self.n_epochs,
            validation_data=(final_test_X, final_test_Y),
        )
        tuner.results_summary()

        model = tuner.get_best_models(num_models=1)[0]
        best_hyperparams = tuner.get_best_hyperparameters()[0]
        log.info(best_hyperparams)
        history = model.fit(
            final_train_X,
            final_train_Y,
            epochs=self.n_epochs,
            batch_size=self.n_batch,
            verbose=1,
            shuffle=True,  # TODO test shuffle
            # callbacks=[es, tensorboard_callback],
            # validation_split=self.validation_split,
            # validation_data=(final_test_X[:-4], final_test_Y[:-4]),
            validation_data=(final_test_X, final_test_Y),
        )

        # if self.debug_plots:
        if True:
            plt.close("all")
            plt.plot(history.history["loss"], color="blue", linestyle="solid", label="Train Set")
            plt.plot(
                history.history["val_loss"],
                color="green",
                linestyle="solid",
                label="Validation Set",
            )
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            output_path = get_run_artifact_path("01", RunArtifact.FORECAST_LOSS)
            plt.savefig(output_path, bbox_inches="tight")
            plt.close("all")

        return model, history, tuner

    def get_X_Y(self, sample_list, label):
        PREDICT_VAR = self.predict_variable + self.scaled_variable_suffix
        X_train_list = list()
        Y_train_list = list()
        df_list = list()

        for i in range(len(sample_list)):
            df = sample_list[i]
            # if label == 'future':
            df_list.append(df)
            df = df.filter(regex="scaled")

            if label != "future":
                X = df.iloc[
                    : -self.predict_days, :
                ]  # exclude last n entries of df to use for prediction
                Y = df.iloc[-self.predict_days :, :]
                labels = np.array(Y[PREDICT_VAR])
            else:
                X = df  # exclude last n entries of df to use for prediction
                Y = 0
                labels = 0

            if not self.predict_var_input_feature:
                X = X.drop(columns=PREDICT_VAR)

            n_rows_train = X.shape[0]
            n_rows_to_add = self.sequence_length - n_rows_train
            pad_rows = np.empty((n_rows_to_add, X.shape[1]), float)
            pad_rows[:] = self.mask_value
            padded_train = np.concatenate((pad_rows, X))

            X_train_list.append(padded_train)
            Y_train_list.append(labels)

        # MAYBE UNCOMMENT NATASHA
        final_test_X = np.array(X_train_list)
        final_test_Y = np.array(Y_train_list)
        final_test_Y = np.squeeze(final_test_Y)
        return final_test_X, final_test_Y  # , df_list

    def create_samples(self, df):
        df_list = list()
        for index in range(len(df.index) + 1):
            i = index
            if (
                i < self.predict_days + self.min_number_of_days
            ):  # only keep df if it has min number of entries
                continue
            else:
                if self.sample_train_length == -1:  # use all historical data for every sample
                    df_list.append(df[:i].copy())
                else:  # use only SAMPLE_LENGTH historical days of data
                    df_list.append(df[i - self.sample_train_length : i].copy())
        return df_list


class MyHyperModel(HyperModel):
    def __init__(
        self, train_sequence_length, predict_sequence_length, n_features, mask_value, batch_size
    ):
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.n_features = n_features
        self.mask_value = mask_value
        self.batch_size = batch_size

    def build(self, hp=None, tune=True, n_layers=-1, dropout=-1, n_hidden_layer_dimensions=-1):
        if tune:
            # access hyperparameters from hp
            dropout = hp.Float("dropout", min_value=0, max_value=0.3, step=0.05, default=0)
            n_hidden_layer_dimensions = hp.Int(
                "n_hidden_layer_dimensions", min_value=10, max_value=100, step=5, default=100
            )
            n_layers = hp.Int("n_layers", min_value=2, max_value=6, step=1, default=4)
            # n_batch = hp.Choice('n_batch', values=[10]) #TODO test other values

        model = Sequential()
        model.add(
            Masking(
                mask_value=self.mask_value,
                batch_input_shape=(self.batch_size, self.train_sequence_length, self.n_features),
            )
        )
        for i in range(n_layers - 1):
            model.add(
                LSTM(
                    n_hidden_layer_dimensions,
                    batch_input_shape=(
                        self.batch_size,
                        self.train_sequence_length,
                        self.n_features,
                    ),
                    activation="sigmoid",
                    stateful=True,
                    return_sequences=True,
                )
            )
        model.add(
            LSTM(
                n_hidden_layer_dimensions,
                batch_input_shape=(self.batch_size, self.train_sequence_length, self.n_features),
                activation="sigmoid",
                stateful=True,
            )
        )
        model.add(Dropout(dropout))
        model.add(Dense(self.predict_sequence_length, activation="sigmoid"))
        es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=3)
        model.compile(loss=smape, optimizer="adam", metrics=["mae", "mape"])

        return model


def smape(y_true, y_pred):
    return 100 * K.mean(abs(y_true - y_pred) / (abs(y_true) + abs(y_pred)) / 2)


def slim(dataframe_input, variables):
    if type(dataframe_input) == list:
        df_list = []
        for df in dataframe_input:
            df_list.append(df[variables].copy())
        return df_list
    else:  # assume there is one inputdataframe
        return dataframe_input[variables].copy()


def rmse(prediction, data):
    error_sum = 0
    prediction_np = np.squeeze(np.asarray(prediction))
    for i, j in zip(prediction_np, data):  # iterate over samples
        for k, l in zip(i, j):  # iterate over predictions for a given sample
            error_sum += abs(k - l)
    return error_sum


def get_aggregate_errors(X, Y, model, scalers_dict, predict_variable, sequence_length):
    forecast = model.predict(X)
    keras_error = tf.keras.losses.MAE(forecast, Y)
    avg_keras_error = sum(keras_error) / len(keras_error)

    sample_errors = []
    unscaled_sample_errors = []
    map_errors = []
    for i, j in zip(Y, forecast):  # iterate over samples
        error_sum = 0
        unscaled_error_sum = 0
        if sequence_length > 1:
            for k, l in zip(i, j):  # iterate over seven days
                error_sum += abs(k - l)
                unscaled_k = scalers_dict[predict_variable].inverse_transform(k.reshape(1, -1))
                unscaled_l = scalers_dict[predict_variable].inverse_transform(l.reshape(1, -1))
                unscaled_error_sum += abs(unscaled_k - unscaled_l)
        else:
            error_sum += abs(i - j)
            unscaled_i = scalers_dict[predict_variable].inverse_transform(i.reshape(1, -1))
            unscaled_j = scalers_dict[predict_variable].inverse_transform(j.reshape(1, -1))
            unscaled_error_sum += abs(unscaled_i - unscaled_j)
            mape = 100 * (abs(unscaled_i - unscaled_j) / unscaled_i)
        map_errors.append(mape)
        sample_errors.append(error_sum)
        unscaled_sample_errors.append(unscaled_error_sum)

    total_unscaled_error = sum(unscaled_sample_errors)
    average_unscaled_error = total_unscaled_error / len(unscaled_sample_errors)

    scaled_error = sum(sample_errors) / (len(sample_errors))

    average_mape = sum(map_errors) / (len(map_errors))
    return (
        float(scaled_error),
        float(total_unscaled_error),
        float(average_unscaled_error),
        float(average_mape),
    )


def align_time_series(series_a, series_b):
    """
  Identify the optimal time shift between two data series based on
  maximal cross-correlation of their derivatives.

  Parameters
  ----------
  series_a: pd.Series
  Reference series to cross-correlate against.
  series_b: pd.Series
  Reference series to shift and cross-correlate against.

  Returns
  -------
  shift: int
  A shift period applied to series b that aligns to series a
  """
    shifts = range(-21, 21)
    valid_shifts = []
    xcor = []
    np.random.seed(42)  # Xcor has some stochastic FFT elements.
    _series_a = np.diff(series_a)

    for i in shifts:
        series_b_shifted = np.diff(series_b.shift(i))
        valid = ~np.isnan(_series_a) & ~np.isnan(series_b_shifted)
        if len(series_b_shifted[valid]) > 0:
            xcor.append(signal.correlate(_series_a[valid], series_b_shifted[valid]).mean())
            valid_shifts.append(i)
    if len(valid_shifts) > 0:
        return valid_shifts[np.argmax(xcor)]
    else:
        return 0


def plot_percentile_error(train_data, test_data, train_metric, test_metric, label, predict_var):
    plt.close("all")
    plt.scatter(
        train_data, train_metric, label="Train", s=2, marker="*",
    )
    plt.scatter(
        test_data, test_metric, label="Test", s=2, marker="*",
    )
    plt.legend()
    plt.xlabel(predict_var)
    plt.ylabel(label)
    output_path = get_run_artifact_path("01", RunArtifact.PERCENTILE_PLOT)
    plt.savefig(output_path + "_" + label + "_scatter.pdf", bbox_inches="tight")

    plt.close("all")
    df = pd.DataFrame({"value": test_data, "metric": test_metric})

    cut = pd.cut(df.value, [0, 20, 50, 100, 200, 300, 500, 1000, 2000, 4000, 6000],)
    boxdf = df.groupby(cut).apply(lambda df: df.metric.reset_index(drop=True)).unstack(0)
    counts = df.groupby(cut).agg(["mean", "median", "count"])
    print(counts)
    ax = sns.boxplot(data=boxdf)
    plt.xticks(rotation=30, fontsize=10)
    # plt.ylim(0, 100)
    plt.xlabel(predict_var)
    plt.ylabel(label)
    fig = ax.get_figure()
    output_path = get_run_artifact_path("01", RunArtifact.PERCENTILE_PLOT)
    plt.savefig(output_path + "_" + label + ".pdf", bbox_inches="tight")
    ax = sns.swarmplot(data=boxdf, color=".25")
    plt.savefig(output_path + "_" + label + "_swarm.pdf", bbox_inches="tight")
    df.to_csv(output_path + "_" + label + "_df.csv")
    return


def smape_array(actual_array, predicted_array):
    array = []
    for i, j in zip(actual_array, predicted_array):
        array.append(100 * (abs(i - j) / (abs(i) + abs(j)) / 2))
    return array


def get_error_metrics(data, prediction):
    mae = tf.keras.losses.MAE(data, prediction)
    mape = tf.keras.losses.MAPE(data, prediction)
    smape = smape_array(data, prediction)
    return np.squeeze(mae), np.squeeze(mape), np.squeeze(smape)


def external_run_forecast():
    ForecastRt.run_forecast()
