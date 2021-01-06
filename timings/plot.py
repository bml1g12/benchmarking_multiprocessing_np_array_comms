"""Plots graphs of timings"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """Saves plots of benchmarks to disk"""
    io_df = pd.read_csv("benchmark_timings_iolimited.csv")
    cpu_df = pd.read_csv("benchmark_timings_cpulimited.csv")

    def plot(df, title):
        """plots graphs of timings"""
        df["groupname"] = df.groupname.str.split("_benchmark", expand=True)[0]
        sns.set(font_scale=1.30)

        def barplot_err(x, y, xerr=None, yerr=None, data=None, **kwargs):
            """Plot a bar graph with hand defined symmetrical error bars"""

            _data = []
            for _i in data.index:

                _data_i = pd.concat([data.loc[_i:_i]] * 3, ignore_index=True, sort=False)
                _row = data.loc[_i]
                if xerr is not None:
                    _data_i[x] = [_row[x] - _row[xerr], _row[x], _row[x] + _row[xerr]]
                if yerr is not None:
                    _data_i[y] = [_row[y] - _row[yerr], _row[y], _row[y] + _row[yerr]]
                _data.append(_data_i)

            _data = pd.concat(_data, ignore_index=True, sort=False)

            _ax = sns.barplot(x=x, y=y, data=_data, ci="sd", **kwargs)

            return _ax

        _, ax = plt.subplots(figsize=(20, 10))
        _ax = barplot_err(x="groupname", y="time_for_all_frames", yerr="stddev_for_all_frames",
                          capsize=.2, data=df, ax=ax)
        for _, row in df.iterrows():
            _ax.text(row.name, row.time_for_all_frames - row.time_for_all_frames * 0.5,
                     f"{round(row.fps, 2)} FPS", color="black", ha="center", va="bottom")

        plt.xlabel(title)
        plt.ylabel("Time to process 1000 frames (s)")
        plt.savefig(title + ".png")

    plot(io_df, "IO_Limited")
    plot(cpu_df, "CPU_Limited")


if __name__ == "__main__":
    main()
