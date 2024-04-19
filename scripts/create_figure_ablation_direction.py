#
import sys

sys.path.append("../")
sys.path.append("./")
import numpy as np
import os

import scripts.helpers as script_help

import pandas as pd

pd.set_option("display.max_columns", None)

import causal_nf.utils.dataframe as causal_nf_df

import matplotlib.pyplot as plt

from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
# plt.rcParams.update(figsizes.icml2022_full())

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt

root = "output_causal_nf"
folder = os.path.join("results", "images")

dataset_name = "CHAIN-4[LIN]"

keep_cols = []
keep_cols.append("dataset__name")
keep_cols.append("dataset__sem_name")
keep_cols.append("dataset__num_samples")
keep_cols.append("dataset__base_version")

keep_cols.append("model__name")
keep_cols.append("model__layer_name")
keep_cols.append("model__dim_inner")
keep_cols.append("model__adjacency")
keep_cols.append("model__base_to_data")
keep_cols.append("model__base_distr")

keep_cols.append("train__regularize")

# %% Load dataframes
df_all = []
for exp_folder in ["ablation_u_x", "ablation_x_u"]:
    df = script_help.load_df(root, [exp_folder], keep_cols, freq=10)
    df_all.append(df.last)

df = pd.concat(df_all, axis=0)

# %%

filter_ = {}

# filter_['optim__base_lr'] = [1e-3]
# filter_['dataset__scale'] = ['default']
# filter_['dataset__sem_name'] = ['linear']


filter_["split"] = ["test"]

# filter_['model__dim_inner'] = [ '[32, 32]' ]
# filter_['train__regularize'] = [ True ]
# filter_['optim__factor'] = [ 0.95 ]


df_ = causal_nf_df.filter_df(df.copy(), filter_)
df_["kl_forward"] = df_["log_prob_true"] - df_["log_prob"]

df_tmp = script_help.update_names(df_)

df_tmp["rmse_cf"] = df_tmp.filter(regex="rmse_cf").mean(1)
df_tmp["mmd_int"] = df_tmp.filter(regex="mmd_int").mean(1)
df_tmp["rmse_ate"] = df_tmp.filter(regex="rmse_ate").mean(1)

df_tmp["loss_jacobian_x"] = (
    df_tmp["loss_jacobian_x"] + np.random.rand(len(df_)) * 0.000000001
)

# %%

cols = []
cols.append("Dataset")
cols.append("log_prob_true")

df_log_prob = df_tmp[cols].groupby(["Dataset"]).agg(["mean", "std"])

# %%

from tueplots import bundles

plt.rcParams.update(bundles.icml2022())
from tueplots import fontsizes

double_ = {}
for key, value in fontsizes.icml2022().items():
    double_[key] = 2.0 * value
plt.rcParams.update(double_)
fontsizes.icml2022()

# %%


x = "$L$"

df_plot = df_tmp.copy()
mapping = {}
jac_loss = r"$\mathcal{L} \left( \nabla_\mathbf{x} T_{\theta}(\mathbf{x}) \right)$"

mapping["loss_jacobian_x"] = jac_loss
mapping["kl_forward"] = "KL forward"
mapping["rmse_ate"] = "RMSE ATE"

df_plot = df_plot.rename(columns=mapping)

df_plot = df_plot[df_plot.Dataset == dataset_name]
y_list = list(mapping.values())

for direction in ["x-u", "u-x"]:
    for y_name, y in mapping.items():
        fig, ax = plt.subplots()
        filename = f"ablation_{direction}_{y_name}"
        for model_name, df_grouped in df_plot[df_plot.Direction == direction].groupby(
            ["Model"]
        ):
            color = script_help.select_color(model_name)
            x_ticks = sorted(df_grouped[x].unique())
            df_grouped[x] = df_grouped[x].map(
                {x_ticks[i]: i for i in range(len(x_ticks))}
            )
            linestyle = script_help.select_style(direction)
            marker = script_help.select_marker(model_name)
            sns.lineplot(
                data=df_grouped,
                x=x,
                y=y,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor="white" if marker == "*" else color,
                markersize=15 if marker == "*" else None,
                markeredgewidth=1.0 if marker == "*" else None,
            )
        ax.grid(True)
        ax.set_xticks(list(range(len(x_ticks))))
        # Apply the formatting function to the y-axis
        ax.yaxis.set_major_formatter(script_help.ticks_formatter[y_name])

        if y_name == "kl_forward":
            ax.set_yscale("log")
        #             ax.set_ylim((None, 0.25))
        ax.set_xticklabels(x_ticks)
        data_name = script_help.remove_non_alphanumeric(dataset_name)
        path = os.path.join(folder, f"{filename}_{data_name}.{script_help.ext}")

        print(f"Saving figure: {path}")
        plt.tight_layout()
        fig.savefig(path)
        plt.close("all")
        # plt.show()

# %%


# For the legend
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

folder = os.path.join("images", "results")
# Create a figure and axis
fig, ax = plt.subplots()

lines = []

models = ["MAF", "MAF*", "CausalMAF", "CausalMAF*"]
m_dict = {}
m_dict["MAF"] = r"Ordering"
m_dict["MAF*"] = r"Ordering$^\star$"
m_dict["CausalMAF"] = r"Graph"
m_dict["CausalMAF*"] = r"Graph$^\star$"

# Create dummy lines for legend without colors
dummy_line_1 = Line2D([], [], linestyle="", color="k", label="Model:")
lines.append(dummy_line_1)
for m in models:
    marker = script_help.select_marker(m)
    color = script_help.select_color(m.replace("*", ""))
    dummy_line_1 = Line2D(
        [],
        [],
        lw=3,
        linestyle="-",
        marker=marker,
        color=color,
        label=m_dict[m],
        markerfacecolor=color,
        markeredgecolor="white" if marker == "*" else color,
        markersize=15 if marker == "*" else 8,
        markeredgewidth=1.0 if marker == "*" else None,
    )
    lines.append(dummy_line_1)

# Create the legend without colors
ax.legend(handles=lines, ncols=5, frameon=False)

# Remove unnecessary plot elements
ax.axis("off")
plt.tight_layout()
# Save the legend as an image file
plt.savefig(os.path.join(folder, "ablation_legend_color.pdf"))

# Show the legend
# plt.show()


# Create a figure and axis
fig, ax = plt.subplots()
lines = []

# Create dummy lines for legend without colors
dummy_line_1 = Line2D([], [], linestyle="", color="k", label="Direction:")
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.3,
    linestyle="-",
    color="k",
    label=r"$\mathbf{x} \rightarrow \mathbf{u}$",
)
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.3,
    linestyle=":",
    color="k",
    label=r"$\mathbf{u} \rightarrow \mathbf{x}$",
)
lines.append(dummy_line_1)

# Create the legend without colors
ax.legend(handles=lines, ncols=3, frameon=False)

# Remove unnecessary plot elements
ax.axis("off")
plt.tight_layout()
# Save the legend as an image file
plt.savefig(os.path.join(folder, "ablation_legend_style.pdf"))

# Show the legend
# plt.show()


# Create a figure and axis
fig, ax = plt.subplots()
lines = []

# Create dummy lines for legend without colors
dummy_line_1 = Line2D([], [], linestyle="", color="k", label="Marker")
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.5,
    marker=script_help.select_marker("MAF"),
    linestyle="-",
    color="k",
    label="No reg.",
)
lines.append(dummy_line_1)
dummy_line_1 = Line2D(
    [],
    [],
    lw=1.5,
    marker=script_help.select_marker("MAF*"),
    linestyle="-",
    color="k",
    label="Reg.",
)
lines.append(dummy_line_1)

# Create the legend without colors
ax.legend(handles=lines, ncols=3, frameon=False)

# Remove unnecessary plot elements
ax.axis("off")
plt.tight_layout()
# Save the legend as an image file
plt.savefig(os.path.join(folder, "ablation_legend_marker.pdf"))

# Show the legend
# plt.show()

plt.close("all")
