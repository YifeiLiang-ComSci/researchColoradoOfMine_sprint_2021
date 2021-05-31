import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, MaxNLocator
import tikzplotlib
import csv
import numpy as np

def get_SNP_name_sequence_dict(path_to_csv):
    """
    
    Examples
    --------
    SNP_label_dict = helpers.get_SNP_name_sequence_dict('./data/snp_labels.csv')
    """
    with open(path_to_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line_count = 0
        for row in csv_reader:
            row_ = row
            line_count += 1
    assert(line_count == 1)
    result_dct = {}
    for i in range(len(row_)):
        result_dct[row_[i]] = i
    return result_dct

def get_adjacency_matrix_from_pairwise_value(SNP_label_dict, SNP_value_df, col_names_pair_vertex_lst, col_name_pairwise_value = 'r^2', threshold_pairwise_value = 0.2):
    assert(len(col_names_pair_vertex_lst) == 2)
    columns_ = SNP_value_df.columns.values
    assert(col_names_pair_vertex_lst[0] in columns_ and col_names_pair_vertex_lst[1] in columns_ and col_name_pairwise_value in columns_)
    number_of_vertices = len(SNP_label_dict)
    adjacency_matrix = np.zeros((number_of_vertices, number_of_vertices))
    for index, row in SNP_value_df.iterrows():
        if row[col_name_pairwise_value] >= threshold_pairwise_value:
            adjacency_matrix[SNP_label_dict[row[col_names_pair_vertex_lst[0]]], SNP_label_dict[row[col_names_pair_vertex_lst[1]]]] = 1
            adjacency_matrix[SNP_label_dict[row[col_names_pair_vertex_lst[1]]], SNP_label_dict[row[col_names_pair_vertex_lst[0]]]] = 1
    return adjacency_matrix

def plot_chromosome(reordered_SNPs_info_df, path_to_save, colors = ["red", "navy", "lightgreen", "lavender", "khaki", "teal", "gold", "violet", "green", "orange", "blue", "coral", "azure", "yellowgreen", "sienna", "olive", "maroon", "goldenrod", "darkblue", "orchid", "crimson"], tick_indices = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 17, 19, 20, 21]):
    reordered_SNPs_info_df_copied = reordered_SNPs_info_df.copy()
    reordered_SNPs_info_df_copied.sort_values(by = ["chr", "weights"], ascending = [True, True])
    # reordered_SNPs_info_df_copied['color'] = reordered_SNPs_info_df_copied["chr"].apply(lambda x: colors[x])
    ax = reordered_SNPs_info_df_copied.plot.scatter(x = 'index_original', y = 'weights', c = 'color', s = 0.03, figsize = (8, 2), colorbar = False, fontsize = 6, marker = ',')
    counts_snps = [0]
    for chr in range(1, 22):
        series_obj = reordered_SNPs_info_df_copied.apply(lambda x: True if x['chr'] == chr else False, axis = 1)
        counts_snps.append(len(series_obj[series_obj == True].index))
    hori_labels = []
    accumulated = 0
    for i in range(21):
        accumulated += counts_snps[i]
        hori_labels.append(round(counts_snps[i + 1] / 2 + accumulated))
    x_ticks_texts = list(range(1, 22))
    x_ticks_colors = deepcopy(colors)
    x_ticks_indices = list(np.array(tick_indices) - 1)

    hori_labels = delete_items_from_list_with_indices(hori_labels, x_ticks_indices, keep_not_remove = True)
    x_ticks_texts = delete_items_from_list_with_indices(x_ticks_texts, x_ticks_indices, keep_not_remove = True)
    x_ticks_colors = delete_items_from_list_with_indices(x_ticks_colors, x_ticks_indices, keep_not_remove = True)

    x_axis = ax.axes.get_xaxis()
    # x_axis.set_visible(False)
    ax.set_xticks(hori_labels)
    ax.set_xticklabels(x_ticks_texts)
    # ax.tick_params(axis = 'x', colors = x_ticks_colors)
    for i in range(len(x_ticks_colors)):
        ax.get_xticklabels()[i].set_color(x_ticks_colors[i])
    ax.margins(x = 0)
    ax.margins(y = 0)
    plt.xlabel("")
    plt.ylabel("Weights")
    # cbar = plt.colorbar(mappable = ax)
    # cbar.remove()
    # plt.margins(y=0)
    # plt.tight_layout()
    # plt.grid(which = 'major', linestyle='-', linewidth=2)
    plt.savefig(path_to_save, bbox_inches = "tight")
    tikzplotlib.save(format_tex_path(path_to_save))

def plot_SNP(reordered_SNPs_info_df, path_to_save : str, bar_width = 'auto', opacity = 0.8, format = 'eps', xticks_fontsize = 6, diagonal_xtickers = False):
    """
    
    Parameters
    ----------
        name_numbers : dict
            For example, name_numbers['enriched'] == [0.12, 0.43, 0.12] for RMSE
        xlabels : list
            Name of groups, For example ['CNN', 'LR', 'SVR', ..]
    """

    reordered_SNPs_info_df_copied = reordered_SNPs_info_df.copy()
    reordered_SNPs_info_df_copied = reordered_SNPs_info_df_copied.sort_values(by = 'weights', ascending = False)
    top_20_SNPs_names = list(reordered_SNPs_info_df_copied.loc[:, "SNP"][:20])
    top_20_SNPs_weights = list(reordered_SNPs_info_df_copied.loc[:, "weights"][:20])
    top_20_SNPs_colors = list(reordered_SNPs_info_df_copied.loc[:, "color"][:20])
    fig = plt.figure(figsize = (7, 4))

    n_groups = 10

    if bar_width == 'auto':
        bar_width_ = 0.1

    ## create plot
    ax_1 = plt.subplot(2, 1, 1)
    index = np.arange(n_groups)

    ## set range
    min_, max_ = np.min(top_20_SNPs_weights), np.max(top_20_SNPs_weights)
    plt.ylim([0.5 * min_, 1.2 * max_])

    rects_list = []
    plt.bar(np.arange(10), top_20_SNPs_weights[:10], alpha = opacity, color = top_20_SNPs_colors[:10])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[:10])
    # plt.legend()
    plt.title('Top-20 SNPs')
    for obj in ax_1.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    
    ax_2 = plt.subplot(2, 1, 2)
    plt.ylim([0.5 * min_, 1.2 * max_])
    plt.bar(np.arange(10), top_20_SNPs_weights[10:], alpha = opacity, color = top_20_SNPs_colors[10:])
    plt.xlabel("")
    plt.ylabel("Weights")
    if diagonal_xtickers:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:], rotation = 45, ha = "right")
    else:
        plt.xticks(index + (bar_width_/2) * (1-1), top_20_SNPs_names[10:])
    for obj in ax_2.get_xticklabels():
        obj.set_fontsize(xticks_fontsize)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    plt.savefig(path_to_save, format = format)
    tikzplotlib.save(format_tex_path(path_to_save))