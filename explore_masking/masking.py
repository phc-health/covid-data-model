import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# import plotly.figure_factory as ff
import json
from urllib.request import urlopen
import plotly.express as px


# OUTPUT DIR
OUTPUT_DIR = "output/"

# DEFAULT PARAMETERS
SMOOTHING_WINDOW_SIZE = 7

# INPUT DATA
MAP_DATA = "input_data/counties.json"
CAN_DATA = "input_data/timeseries.csv"
MASKING_DATA = "input_data/masking_data.csv"
WEIGHTED_MASKING_DATA = "input_data/w_masking_data.csv"
RT_DATA = "input_data/rt_combined_metric.csv"


def make_scatter_plot(df, var1, var2):
    fig = plt.figure()
    plt.scatter(df[var1], df[var2], s=0.5)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[var1], df[var2])
    print(
        f"slope: {slope:.1f} intercept: {intercept} r_value: {r_value} p_value: {p_value} std_err: {std_err}"
    )
    y = slope * df[var1] + intercept

    plt.plot(
        df[var1],
        y,
        label=f"Lin. Reg. y = {slope:.1f}*x + {intercept:.1f} \n r_value: {r_value:.1f} p_value: {p_value:.1f} std_err: {std_err:.1f}",
    )
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.savefig(OUTPUT_DIR + var1 + "_" + var2 + ".pdf")


def make_box_plots(df, var1, var2, binning):
    fig = plt.figure()
    df["binned_value"] = pd.cut(df[var1], binning)
    # ax = sns.boxplot(x=df["binned_value"], y=df[var2])
    plt.xticks(rotation=30, fontsize=10)
    ax = sns.violinplot(x=df["binned_value"], y=df[var2], color=".25")
    # ax = sns.swarmplot(x=df["binned_value"], y=df[var2], color=".25")
    plt.xlabel(var1)
    plt.ylabel(var2)
    fig = ax.get_figure()
    # plt.savefig(OUTPUT_DIR + "_" + var1 + "_" + var2 + ".pdf", bbox_inches="tight")

    # plt.savefig(OUTPUT_DIR + "_" + var1 + "_" + var2 + "_swarm.pdf", bbox_inches="tight")

    plt.savefig(OUTPUT_DIR + "_" + var1 + "_" + var2 + "_violin.pdf", bbox_inches="tight")
    return


def process_raw_df(df):
    df["new_cases"] = df["cases"].diff()
    df["new_cases_smooth"] = df["new_cases"].rolling(SMOOTHING_WINDOW_SIZE).mean()
    return df


def make_map_plot(df, var, date):
    df = df[df["date"] == date]
    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        counties = json.load(response)
    fig = px.choropleth(
        df,
        geojson=counties,
        locations="fips",
        color=var,
        color_continuous_scale="Viridis",
        range_color=(0, 1000),
        scope="usa",
        labels={var: var},
    )
    fig.show()
    fig.save("test.pdf")
    # fig.write_image(OUTPUT_DIR + "map_" + var + ".pdf")
    print("MADE")


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
    masking_df = masking_df[masking_df["masking_percentage"] > 80]
    df = pd.merge(
        can_df,
        masking_df,
        how="inner",
        left_on=["fips", "date"],
        right_on=["geo_value", "time_value"],
    )
    make_scatter_plot(df, "masking_percentage", "new_cases_smooth")
    make_scatter_plot(df, "masking_percentage", "Rt_MAP__new_cases")
    make_scatter_plot(df, "weighted_masking_percentage", "new_cases_smooth")
    make_scatter_plot(df, "weighted_masking_percentage", "Rt_MAP__new_cases")
    make_scatter_plot(df, "masking_percentage", "weighted_masking_percentage")
    exit()

    new_cases_binning = [0, 50, 100, 300, 500, 1000, 3000]
    masking_binning = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    rt_binning = [0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2]
    make_box_plots(df, "masking_percentage", "new_cases_smooth", masking_binning)
    make_box_plots(df, "masking_percentage", "Rt_MAP__new_cases", masking_binning)
    make_box_plots(df, "weighted_masking_percentage", "new_cases_smooth", masking_binning)
    make_box_plots(df, "weighted_masking_percentage", "Rt_MAP__new_cases", masking_binning)
    # reg = LinearRegression().fit(np.array(df['masking_percentage']).reshape(-1,1), np.array(df['masking_percentage']).reshape(-1,1) )
    # reg.score(df['masking_percentage'], df['masking_percentage'])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["masking_percentage"], df["new_cases_smooth"]
    )
    print(
        f"slope: {slope} intercept: {intercept} r_value: {r_value} p_value: {p_value} std_err: {std_err}"
    )
    # make_map_plot(df, "sample_size_x", "2020-09-13")
