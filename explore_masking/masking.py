import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#OUTPUT DIR
OUTPUT_DIR = "output/"

#DEFAULT PARAMETERS
SMOOTHING_WINDOW_SIZE = 7

#INPUT DATA
CAN_DATA = "input_data/timeseries.csv"
MASKING_DATA = "input_data/masking_data.csv"
WEIGHTED_MASKING_DATA = "input_data/w_masking_data.csv"
RT_DATA = "input_data/rt_combined_metric.csv"


def make_scatter_plot(df, var1, var2):
  fig = plt.figure()
  plt.scatter(df[var1], df[var2], s = 0.5)
  plt.xlabel(var1)
  plt.ylabel(var2)
  plt.savefig(OUTPUT_DIR + var1 + "_" + var2 + '.pdf')

def make_box_plots(df, label, predict_var, dataset_name):
    plt.close("all")
    metric_binning = [0, 20, 50, 100, 200, 300, 500, 1000, 3000]
    cut = pd.cut(df.value, metric_binning)
    boxdf = df.groupby(cut).apply(lambda df: df.metric.reset_index(drop=True)).unstack(0)
    # ax = sns.boxplot(data=boxdf)
    ax = sns.boxplot(x=df.binned_value, y=df.metric)
    plt.xticks(rotation=30, fontsize=10)
    plt.title(dataset_name)
    if label == "mae":
        plt.ylim(0, 500)
    plt.xlabel(predict_var)
    plt.ylabel(label)
    fig = ax.get_figure()
    output_path = get_run_artifact_path("01", RunArtifact.PERCENTILE_PLOT)
    plt.savefig(output_path + "_" + label + "-" + dataset_name + ".pdf", bbox_inches="tight")
    ax = sns.swarmplot(data=boxdf, color=".25")
    plt.savefig(output_path + "_" + label + "-" + dataset_name + "_swarm.pdf", bbox_inches="tight")
    plt.close("all")
    return

def process_raw_df(df):
  df['new_cases'] = df['cases'].diff()
  df['new_cases_smooth'] = df['new_cases'].rolling(SMOOTHING_WINDOW_SIZE).mean()
  return df

if __name__ == '__main__':
  #MAKE DATAFRAMES
  raw_df = pd.read_csv(CAN_DATA)
  raw_df = process_raw_df(raw_df)
  rt_df = pd.read_csv(RT_DATA)

  raw_masking_df = pd.read_csv(MASKING_DATA)
  raw_masking_df.rename(columns={'value': 'masking_percentage'}, inplace=True)
  weighted_masking_df = pd.read_csv(WEIGHTED_MASKING_DATA)
  weighted_masking_df.rename(columns={'value': 'weighted_masking_percentage'}, inplace=True)

  #MERGE DATAFRAMES
  can_df = pd.merge(raw_df, rt_df, how = 'inner', left_on = ['fips', 'date'], right_on = ['fips', 'date']) 
  masking_df = pd.merge(raw_masking_df, weighted_masking_df, how = 'inner', on = ['geo_value', 'time_value'])
  print(masking_df)
  print(masking_df.columns)
  df = pd.merge(can_df, masking_df, how = 'inner', left_on = ["fips", "date"], right_on = ['geo_value', 'time_value'])
  make_scatter_plot(df, 'masking_percentage', 'new_cases_smooth')
  make_scatter_plot(df, 'masking_percentage', 'Rt_MAP__new_cases')

