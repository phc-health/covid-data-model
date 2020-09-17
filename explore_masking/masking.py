import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# import plotly.figure_factory as ff
# import json
# from urllib.request import urlopen
import geopandas
import json


# OUTPUT DIR
OUTPUT_DIR = "output/"

# DEFAULT PARAMETERS
SMOOTHING_WINDOW_SIZE = 7

# INPUT DATA
MAP_DATA = "input_data/geojson-counties-fips.json"
CAN_DATA = "input_data/timeseries.csv"
MASKING_DATA = "input_data/masking_data.csv"
WEIGHTED_MASKING_DATA = "input_data/w_masking_data.csv"
FB_SURVEY = "input_data/fb_raw_cli.csv"
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


def make_map_plot(df, var, date, var_min, var_max):
    counties = geopandas.read_file(MAP_DATA)
    counties["fips"] = counties["id"].astype(int)
    counties_df = pd.DataFrame(counties)

    df = df[df["date"] == date]
    n_entries = df["fips"].count()
    n_unique = df["fips"].nunique()
    print(f"var: {var} entries: {n_entries} unique entries: {n_unique}")
    merged_df = pd.merge(df, counties, how="outer", on="fips").fillna(np.NaN)
    merged_df["test_var"] = -10

    merged_df = merged_df[
        (merged_df["STATE"] != "15") & (merged_df["STATE"] != "02") & (merged_df["STATE"] != "72")
    ]
    merged_df.to_csv("merged.csv")
    combined_geo = geopandas.GeoDataFrame(merged_df)

    fig, ax = plt.subplots(1, 1)
    my_cmap = plt.cm.get_cmap("jet")
    my_cmap.set_under("grey")
    # my_cmap.set_over("magenta")
    plt.rc("font", size=40)

    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=var_min, vmax=var_max))
    ax = combined_geo.plot(
        column="test_var", figsize=(60, 40), cmap=my_cmap, vmin=var_min, vmax=var_max
    )
    plt.title(var)
    plt.axis("off")
    fig = ax.get_figure()
    cax = fig.add_axes([0.1, 0.2, 0.8, 0.01])
    fig.colorbar(sm, orientation="horizontal", aspect=80, label=var, cax=cax, extend="both")
    plt.savefig(OUTPUT_DIR + "map_" + var + ".pdf")
    plt.close("all")


if __name__ == "__main__":
    # CAN DATA
    raw_df = pd.read_csv(CAN_DATA)
    raw_df = process_raw_df(raw_df)
    rt_df = pd.read_csv(RT_DATA)

    # MASKING DATA
    raw_masking_df = pd.read_csv(MASKING_DATA)
    raw_masking_df.rename(
        columns={"value": "masking_percentage", "time_value": "date", "geo_value": "fips"},
        inplace=True,
    )
    weighted_masking_df = pd.read_csv(WEIGHTED_MASKING_DATA)
    weighted_masking_df.rename(
        columns={"time_value": "date", "geo_value": "fips", "value": "weighted_masking_percentage"},
        inplace=True,
    )

    fb_cli_df = pd.read_csv(FB_SURVEY)
    fb_cli_df.rename(
        columns={"value": "fb_raw_cli", "time_value": "date", "geo_value": "fips"}, inplace=True
    )

    # MERGE DATAFRAMES
    can_df = pd.merge(raw_df, rt_df, how="inner", on=["fips", "date"])
    masking_df = pd.merge(raw_masking_df, weighted_masking_df, how="inner", on=["fips", "date"])
    # masking_df = masking_df[masking_df["masking_percentage"] > 80]
    df = pd.merge(can_df, masking_df, how="outer", on=["fips", "date"]).fillna(0)

    make_map_plot(df, "masking_percentage", "2020-09-16", 0, 100)
    make_map_plot(fb_cli_df, "fb_raw_cli", "2020-09-16", 0, 5)
    make_map_plot(can_df, "new_cases", "2020-09-15", 0, 3000)

    exit()

    make_scatter_plot(df, "masking_percentage", "new_cases_smooth")
    make_scatter_plot(df, "masking_percentage", "Rt_MAP__new_cases")
    make_scatter_plot(df, "weighted_masking_percentage", "new_cases_smooth")
    make_scatter_plot(df, "weighted_masking_percentage", "Rt_MAP__new_cases")
    make_scatter_plot(df, "masking_percentage", "weighted_masking_percentage")

    new_cases_binning = [0, 50, 100, 300, 500, 1000, 3000]
    masking_binning = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    rt_binning = [0, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2]
    make_box_plots(df, "masking_percentage", "new_cases_smooth", masking_binning)
    make_box_plots(df, "masking_percentage", "Rt_MAP__new_cases", masking_binning)
    make_box_plots(df, "weighted_masking_percentage", "new_cases_smooth", masking_binning)
    make_box_plots(df, "weighted_masking_percentage", "Rt_MAP__new_cases", masking_binning)


def obsolete_make_map_plot(df, var, date):
    df = df[df["date"] == date]
    # with urlopen(
    #    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    # ) as response:
    #    counties = json.load(response)
    fig = px.choropleth(
        df,
        geojson=counties,
        locations="fips",
        color=var,
        color_continuous_scale="Viridis",
        range_color=(0, 3),
        scope="usa",
        labels={var: var},
    )
    fig.show()
    fig.save("test.pdf")
    # fig.write_image(OUTPUT_DIR + "map_" + var + ".pdf")
    print("MADE")
