"""Adapted from https://github.com/zzzace2000/GAMs_models/.

Visualization utilities include plotting GAMs and comparing pandas tables.
"""


import numbers

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def vis_GAM_effects(
    all_dfs,
    # Figure params
    num_cols=4,
    figsize=None,
    vertical_margin=2,
    horizontal_margin=2,
    sort_by_imp=False,
    show_density=False,
    # Filtering
    model_names=None,
    feature_names=None,
    feature_idxes=None,
    top_main=-1,
    top_interactions=-1,
    only_interactions=False,
    # Others
    call_backs=None
):
    """Visualize main and interaction effects of the GAM model.

    Args:
        all_dfs: the dictionary of dataframes. The key is the model name and the value is the GAM
            dataframe of each model.
        num_cols: number of columns when showing GAM graphs.
        figsize: the figure size. If not specified, it uses the
                (width, height) = (4 * num_cols + (num_cols-1) * horizontal_margin,
                                   3 * num_rows + vertical_margin * (num_rows - 1)).
        vertical_margin: the vertical margin. Default: 2.
        horizontal_margin: the horizontal margin. Default: 2.
        sort_by_imp: if True, sort the figures by the feature importances. Otherwise use the feature
            default order.
        show_density: if True, it represents the data density as color red in the background when
            showing the main effect GAM graph.
        model_names: if specified, only show the GAM models corresponding to these.
        feature_names: if specified, only show the GAM graphs corresponding to these names.
        feature_idxes: if specified, only show the GAM graphs corresponding to these feature index.
        top_main: if > 0, only show the top k main effects. If -1, show all main effects.
        top_interactions: if > 0, only show the top k interactions. If -1, show all interactions.
        only_interactions: if True, hide all the main effect plots and only show interaction terms.
        call_backs: if specified, it calls this function at the end of plotting the graph. It
            should be a dict with key as the feature name and the value as a function
            (lambda ax: f(ax)) that can modify the axis corresponding to that feature. Useful to do
            feature-specific adjustment.

    Returns:
        fig: the figure.
        axes (numpy array): all the axes.
    """
    if model_names is None:
        model_names = list(all_dfs.keys())
    else:
        all_dfs = {k: all_dfs[k] for k in model_names}

    first_df = all_dfs[next(iter(all_dfs))]
    first_df = first_df[first_df.feat_idx != -1]  # Remove bias first
    if feature_idxes is not None:
        first_df = first_df[first_df.feat_idx.apply(lambda x: x in feature_idxes)]
        first_df['feat_idx'] = pd.Categorical(first_df['feat_idx'], feature_idxes)
        first_df = first_df.sort_values('feat_idx')
    else:
        if feature_names is not None:
            first_df = first_df[first_df.feat_name.apply(lambda x: x in feature_names)]

        # Handle main effect
        df_main = first_df[first_df.feat_idx.apply(lambda x: not isinstance(x, tuple))]
        if sort_by_imp:
            df_main = df_main.sort_values('importance', ascending=False)
        if top_main >= 0:
            df_main = df_main.iloc[:top_main]

        df_iter = first_df[first_df.feat_idx.apply(lambda x: isinstance(x, tuple))]
        if top_interactions >= 0:
            df_iter = df_iter.sort_values('importance', ascending=False).iloc[:top_interactions]
        first_df = pd.concat([df_main, df_iter], axis=0)

        if only_interactions:
            first_df = first_df[first_df.feat_idx.apply(lambda x: isinstance(x, tuple))]

    num_rows = int(np.ceil((len(first_df)) / num_cols))

    if figsize is None:
        figsize = (4 * num_cols + horizontal_margin * (num_cols - 1),
                   3 * num_rows + vertical_margin * (num_rows - 1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1 and isinstance(axes, np.ndarray) and len(axes.shape) == 1:
        axes = axes[None, :]

    the_df_lookups = {k: df.set_index('feat_idx') for k, df in all_dfs.items()}

    ax_idx = 0
    for r_idx, row in first_df.iterrows():
        fig_idx1, fig_idx2 = (ax_idx // num_cols, ax_idx % num_cols)

        the_ax = axes if not isinstance(axes, np.ndarray) else axes[fig_idx1, fig_idx2]

        if isinstance(row.feat_idx, numbers.Number):  # main effect
            if isinstance(row.x[0], str):  # categorical variable
                y_dfs = []
                yerr_dfs = []
                for model_name in model_names:
                    lookup = the_df_lookups[model_name]
                    y_df = pd.DataFrame(lookup.loc[row.feat_idx].y,
                                        index=lookup.loc[row.feat_idx].x,
                                        columns=[model_name])
                    y_dfs.append(y_df)

                    if 'y_std' not in lookup.loc[row.feat_idx]:
                        continue

                    yerr_df = pd.DataFrame(
                        lookup.loc[row.feat_idx].y_std,
                        index=lookup.loc[row.feat_idx].x,
                        columns=[model_name])
                    yerr_dfs.append(yerr_df)

                y_dfs = pd.concat(y_dfs, axis=1)
                if len(yerr_dfs) > 0:
                    yerr_dfs = pd.concat(yerr_dfs, axis=1)
                else:
                    yerr_dfs = None

                y_dfs.plot.bar(ax=the_ax, yerr=yerr_dfs)

                # Rotate back to 0
                for tick in the_ax.get_xticklabels():
                    tick.set_rotation(0)
            else:
                for model_name in model_names:
                    if model_name not in all_dfs:
                        print('%s not in the all_dfs' % model_name)
                        continue

                    the_df_lookup = the_df_lookups[model_name]
                    if row.feat_idx not in the_df_lookup.index:
                        continue

                    y_std = 0 if 'y_std' not in the_df_lookup.loc[row.feat_idx] \
                        else the_df_lookup.loc[row.feat_idx].y_std

                    the_ax.errorbar(
                        the_df_lookup.loc[row.feat_idx].x,
                        the_df_lookup.loc[row.feat_idx].y,
                        y_std, label=model_name)
            the_ax.legend()

            if show_density and hasattr(row, 'counts'):  # main effect
                def shade_by_density_blocks(n_blocks: int = 40, color: list = [0.9, 0.5, 0.5]):
                    x_n_blocks = min(n_blocks, len(row.x))

                    max_x, min_x = row.x[-1], row.x[0]
                    segments = (max_x - min_x) / x_n_blocks
                    density = np.histogram(row.x, bins=x_n_blocks, weights=row.counts)
                    normed_density = density[0] / np.max(density[0])
                    rect_params = []
                    for p in range(x_n_blocks):
                        start_x = min_x + segments * p
                        end_x = min_x + segments * (p + 1)
                        d = min(1.0, 0.01 + normed_density[p])
                        rect_params.append((d, start_x, end_x))

                    min_y, max_y = the_ax.get_ylim()
                    for param in rect_params:
                        alpha, start_x, end_x = param
                        rect = patches.Rectangle(
                            (start_x, min_y - 1),
                            end_x - start_x,
                            max_y - min_y + 1,
                            linewidth=0.01,
                            edgecolor=color,
                            facecolor=color,
                            alpha=alpha,
                        )
                        the_ax.add_patch(rect)

                shade_by_density_blocks()

        else:  # interaction effect
            all_x = [t[0] for t in row.x]
            all_y = [t[1] for t in row.x]

            fid1, fid2 = row.feat_idx
            tmp_df = all_dfs[next(iter(all_dfs))]
            feat_names = [
                tmp_df[tmp_df.feat_idx == int(fid1)].feat_name.iloc[0],
                tmp_df[tmp_df.feat_idx == int(fid2)].feat_name.iloc[0],
            ]

            x_len, y_len = len(set(all_x)), len(set(all_y))
            if x_len > 4 and y_len > 4:
                cax = sns.scatterplot(x=all_x, y=all_y, hue=row.y, palette='RdBu', ax=the_ax, s=50)
                the_ax.set_xlabel(feat_names[0])
                the_ax.set_ylabel(feat_names[1])

                vlim = np.max(np.abs(row.y))
                norm = plt.Normalize(-vlim, vlim)
                sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
                sm.set_array([])

                # Remove the legend and add a colorbar
                cax.get_legend().remove()
                cax.figure.colorbar(sm, ax=the_ax)

            elif x_len <= 4 and x_len < y_len:
                uniq_x, inv = np.unique(all_x, return_inverse=True)

                for i, x in enumerate(uniq_x):
                    y = np.array(all_y)[inv == i]
                    val = np.array(row.y)[inv == i]
                    val_std = 0.
                    if 'y_std' in row:
                        val_std = np.array(row.y_std)[inv == i]
                    the_ax.errorbar(y, val, val_std, label=f'{feat_names[0]}={x}')
                the_ax.set_xlabel(feat_names[1])
                the_ax.legend()
            else:
                uniq_y, inv = np.unique(all_y, return_inverse=True)

                for i, x in enumerate(uniq_y):
                    y = np.array(all_x)[inv == i]
                    val = np.array(row.y)[inv == i]
                    val_std = 0.
                    if 'y_std' in row:
                        val_std = np.array(row.y_std)[inv == i]
                    the_ax.errorbar(y, val, val_std, label=f'{feat_names[1]}={x}')
                the_ax.set_xlabel(feat_names[0])
                the_ax.legend()

        title = row.feat_name
        if 'importance' in row:
            title = f'{row.feat_name} (Imp={round(row.importance, 3)})'
        the_ax.set_title(title)
        ax_idx += 1

        if call_backs is not None and row.feat_name in call_backs:
            call_backs[row.feat_name](the_ax)

    # Finally, close all the remaining plots
    for idx in range(ax_idx, len(axes.flat)):
        axes.flat[idx].set_axis_off()

    return fig, axes


def cal_statistics(table, is_metric_higher_better=True, add_ns_baseline=False):
    """Calculate the statistics like average, average ranks across scores.

    Args:
        table: a pandas table with each row as method and column as different datasets.
        is_metric_higher_better: if True, treat the higher metric as better.
        add_ns_baseline: if True, add an normalized score to the statistics.

    Returns:
        A pandas table with two summary row as (1) average value, and (2) the average ranks. It
            also highlights the best number as red and the worst method as green.
    """
    # Add two rows
    mean_score = table.apply(lambda x: x.apply(
        lambda s: float(s[:s.index(' +-')] if isinstance(s, str) and ' +-' in s else s)).mean(),
    axis=0)
    new_table = add_new_row(table, mean_score, 'average')
    
    average_rank = mean_score.rank(ascending=(not is_metric_higher_better))
    new_table = add_new_row(new_table, average_rank, 'average_rank')

    mean_rank = table.apply(rank, axis=1, is_metric_higher_better=is_metric_higher_better).mean()
    new_table = add_new_row(new_table, mean_rank.apply(lambda x: '%.2f' % x), 'avg_rank')
    
    avg_rank_rank = mean_rank.rank(ascending=True)
    new_table = add_new_row(new_table, avg_rank_rank, 'avg_rank_rank')

    mean_normalized_score = table.apply(normalized_score, axis=1,
                                        is_metric_higher_better=is_metric_higher_better).mean()
    new_table = add_new_row(new_table, mean_normalized_score.apply(lambda x: '%.3f' % x),
                            'avg_score')
    
    avg_score_rank = mean_normalized_score.rank(ascending=False)
    new_table = add_new_row(new_table, avg_score_rank, 'avg_score_rank')
    
    if add_ns_baseline:
        mean_normalized_score_b = table.apply(
            normalized_score, axis=1, min_value=0.5,
            is_metric_higher_better=is_metric_higher_better).mean()
        new_table = add_new_row(new_table, mean_normalized_score_b.apply(lambda x: '%.3f' % x),
                                'avg_score_b0.5')

        avg_score_rank_b = mean_normalized_score_b.rank(ascending=False)
        new_table = add_new_row(new_table, avg_score_rank_b, 'avg_score_b0.5_rank')

    return new_table


def extract_mean(s):
    """Extract the mean and remove stdev from the content of 0.123 +- 0.234.

    Args:
        s: the string with format "mean +- stdev". E.g. "0.123 +- 0.234".

    Returns:
        mean: a float number, e.g. 0.123.
    """
    if isinstance(s, float):
        return s

    if isinstance(s, str):
        if ' +-' in s:
            s = s[:s.index(' +-')]
        return float(s)

    raise Exception('the input is wierd: %s' % str(s))


def rank(x, is_metric_higher_better=True, is_extract_mean=True):
    if is_extract_mean:
        x = x.apply(extract_mean)
    return x.rank(method='average', ascending=(not is_metric_higher_better), na_option='bottom')


def normalized_score(x, is_metric_higher_better=True, min_value=None):
    x = x.apply(extract_mean)

    if min_value is None:
        min_value = x.min()

    score = (x - min_value) / (x.max() - min_value)
    if not is_metric_higher_better:
        score = 1. - score
    return score


def add_new_row(table, series, row_name):
    new_indexes = list(table.index) + [row_name]
    new_table = table.append(series, ignore_index=True)
    new_table.index = new_indexes
    return new_table


def highlight_min_max(x, is_extract_mean=True):
    if is_extract_mean:
        x = x.apply(extract_mean)
    return ['background-color: #cd4f39' if v == np.nanmax(x)
            else ('background-color: lightgreen' if v == np.nanmin(x) else '') for v in x]

