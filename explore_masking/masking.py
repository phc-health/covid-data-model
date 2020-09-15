import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# OUTPUT DIR
OUTPUT_DIR = "output/"

# DEFAULT PARAMETERS
SMOOTHING_WINDOW_SIZE = 7

# INPUT DATA
CAN_DATA = "input_data/timeseries.csv"
MASKING_DATA = "input_data/masking_data.csv"
WEIGHTED_MASKING_DATA = "input_data/w_masking_data.csv"
RT_DATA = "input_data/rt_combined_metric.csv"


def make_scatter_plot(df, var1, var2):
    fig = plt.figure()
    plt.scatter(df[var1], df[var2], s=0.5)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.savefig(OUTPUT_DIR + var1 + "_" + var2 + ".pdf")


def make_box_plots(df, var1, var2, binning):
    fig = plt.figure()

    df["binned_value"] = pd.cut(df[var1], binning)
    print(df)
    ax = sns.boxplot(x=df["binned_value"], y=df[var2])
    plt.xticks(rotation=30, fontsize=10)
    plt.xlabel(var1)
    plt.ylabel(var2)
    fig = ax.get_figure()
    plt.savefig(OUTPUT_DIR + "_" + var1 + "_" + var2 + ".pdf", bbox_inches="tight")
    ax = sns.swarmplot(x=df["binned_value"], y=df[var2], color=".25")
    plt.savefig(OUTPUT_DIR + "_" + var1 + "_" + var2 + "_swarm.pdf", bbox_inches="tight")
    return


def process_raw_df(df):
    df["new_cases"] = df["cases"].diff()
    df["new_cases_smooth"] = df["new_cases"].rolling(SMOOTHING_WINDOW_SIZE).mean()
    return df


if __name__ == "__main__":
    # MAKE DATAFRAMES
    raw_df = pd.read_csv(CAN_DATA)
    raw_df = process_raw_df(raw_df)
    rt_df = pd.read_csv(RT_DATA)

    raw_masking_df = pd.read_csv(MASKING_DATA)
    raw_masking_df.rename(columns={"value": "masking_percentage"}, inplace=True)
    weighted_masking_df = pd.read_csv(WEIGHTED_MASKING_DATA)
    weighted_masking_df.rename(columns={"value": "weighted_masking_percentage"}, inplace=True)

    # MERGE DATAFRAMES
    can_df = pd.merge(
        raw_df, rt_df, how="inner", left_on=["fips", "date"], right_on=["fips", "date"]
    )
    masking_df = pd.merge(
        raw_masking_df, weighted_masking_df, how="inner", on=["geo_value", "time_value"]
    )
    print(masking_df)
    print(masking_df.columns)
    df = pd.merge(
        can_df,
        masking_df,
        how="inner",
        left_on=["fips", "date"],
        right_on=["geo_value", "time_value"],
    )
    make_scatter_plot(df, "masking_percentage", "new_cases_smooth")
    make_scatter_plot(df, "masking_percentage", "Rt_MAP__new_cases")

    new_cases_binning = [0, 50, 100, 300, 500, 1000, 3000]
    masking_binning = [60, 70, 80, 90, 100]
    rt_binning = [0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2]
    # make_box_plots(df, 'new_cases_smooth', 'masking_percentage', new_cases_binning)
    # make_box_plots(df, 'Rt_MAP__new_cases', 'masking_percentage', rt_binning)
    make_box_plots(df, "masking_percentage", "new_cases_smooth", masking_binning)
    make_box_plots(df, "masking_percentage", "Rt_MAP__new_cases", masking_binning)
