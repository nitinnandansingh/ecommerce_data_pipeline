"""Visualization Utilities."""
import os
import random
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps


def plot_image_samples(
    title: str,
    df: pd.DataFrame,
    data_root: str,
    img_col_name: str,
    label_col_name: str,
    num_labels_to_sample: int = 5,
    num_samples_per_label: int = 5,
    mask_col_name: str = None,
    bbox_col_names: List = [],
    bbox_thickness: int = 5,
    bbox_color: str = "green",
    show_mask: bool = False,
    show_bbox: bool = False,
    sample_fig_width: float = 3,
):
    """Plot image samples.

    Args:
        title (str): Title of the plot.
        df (pd.DataFrame): Data frame containing valid image samples data.
        data_root (str): Root directory path containing image files.
        img_col_name (str): Column name in 'df' corresponding to image file paths which are relative to 'data_root'.
        label_col_name (str): Column name in 'df' corresponding to a label set.
        num_labels_to_sample (int): Number of unique labels to sample. Defaults to 5.
        num_samples_per_label (int, optional): Number of unique samples to sample per unique label. Defaults to 5.
        mask_col_name (str, optional): Column name in 'df' corresponding to mask file paths which are relaitve to
            'data_root'.Defaults to None.
        bbox_col_names (List, optional): List of column names in 'df' corresponding to bounding box cooridnates of
            an object in the sampled image. Defaults to [].
        show_mask (bool, optional): If True, plots mask samples also. Defaults to False.
        show_bbox (bool, optional): If True, plots bounding boxes also. Defaults to False.
        sample_fig_width (float, optional): Desired size of each figure in the plot. Defaults to 3.
    """
    if show_bbox and len(bbox_col_names) == 0:
        raise AssertionError("Column names of bounding box cooridnates must be specified, if show_bbox is True.")

    if show_mask and mask_col_name is None:
        raise AssertionError("Column name of mask file must be specified, if show_mask is True.")

    label_ids = df[label_col_name].sample(frac=1).drop_duplicates().sample(num_labels_to_sample).tolist()
    min_num_samples = num_samples_per_label
    for label_id in label_ids:
        sample_df = df[df[label_col_name] == label_id].copy().reset_index(drop=True)
        min_num_samples = min(min_num_samples, len(sample_df))

    rows = num_labels_to_sample
    if show_mask:
        rows = rows * 2
    cols = min_num_samples

    fig, axs = plt.subplots(
        nrows=rows,
        ncols=1,
        figsize=(cols * sample_fig_width, rows * sample_fig_width),
        facecolor=(1, 1, 1),
        sharey=True,
    )
    fig.suptitle(title, y=1, fontsize=14)
    subplot_idx = 1

    if show_mask:
        label_ids = [label_id for label_id in label_ids for _ in range(2)]

    for i, label_id in enumerate(label_ids):
        if show_mask:
            if i % 2 == 0:
                axs[i].set_title(label_id)
                sample_df = df[df[label_col_name] == label_id].copy().reset_index(drop=True)
                sample_df = sample_df.sample(min_num_samples).reset_index(drop=True)
        else:
            axs[i].set_title(label_id)
            sample_df = df[df[label_col_name] == label_id].copy().reset_index(drop=True)
            sample_df = sample_df.sample(min_num_samples).reset_index(drop=True)

        axs[i].axis("off")
        for j, row in sample_df.iterrows():
            img_file = os.path.join(data_root, row[img_col_name])
            ax = fig.add_subplot(rows, cols, subplot_idx)
            img = Image.open(img_file).convert("RGB")
            img = ImageOps.exif_transpose(img)
            if show_mask:
                if i % 2 == 0:
                    if show_bbox:
                        bbox = [row[bbox_col_name] for bbox_col_name in bbox_col_names]
                        if sum(bbox) != -4:
                            draw = ImageDraw.Draw(img)
                            draw.rectangle(bbox, outline=bbox_color, width=bbox_thickness)
                    ax.imshow(img)  # , aspect="auto")
                else:
                    mask_file = os.path.join(data_root, row[mask_col_name])
                    mask = Image.open(mask_file).convert("RGB")
                    if show_bbox:
                        bbox = [row[bbox_col_name] for bbox_col_name in bbox_col_names]
                        if sum(bbox) != -4:
                            draw = ImageDraw.Draw(mask)
                            draw.rectangle(bbox, outline=bbox_color, width=bbox_thickness)
                    ax.imshow(mask)  # , aspect="auto")
            else:
                if show_bbox:
                    bbox = [row[bbox_col_name] for bbox_col_name in bbox_col_names]
                    if sum(bbox) != -4:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(bbox, outline=bbox_color, width=bbox_thickness)
                ax.imshow(img)  # , aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            subplot_idx += 1
    fig.tight_layout()


def plot_label_distribution(
    title: str,
    df: pd.DataFrame,
    label_col_name: str,
    xlabel: str,
    yscale: str = None,
    show_xticks: bool = False,
    width: int = 16,
    height: int = 3,
    color_palette: str = "RdBu",
):
    """Plot label distribution using bar chart.

    Args:
        title (str): Title of the plot.
        df (pd.DataFrame): Data frame containing valid label data to be plotted.
        label_col_name (str): Column name in 'df' corresponding to label set of interest.
        xlabel (str): Label name for x-axis.
        yscale (str, optional): Scale to be used for y-axis. Defaults to None. Other option is 'log'.
        show_xticks (bool, optional): If True, show label names on x-axis. Defaults to False.
        width (int, optional): Width of the plot. Defaults to 16.
        height (int, optional): Height of the plot. Defaults to 3.
    """
    d = df.groupby(label_col_name).size().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(width, height), sharex=True)
    if xlabel.endswith("s"):
        fig.suptitle("{} ({} {})".format(title, len(d.keys().tolist()), xlabel))
    else:
        fig.suptitle("{} ({} {}s)".format(title, len(d.keys().tolist()), xlabel))
    sns.barplot(x=d.keys().tolist(), y=d.tolist(), palette=color_palette, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if yscale is not None:
        ax.set_yscale(yscale)
    if show_xticks:
        ax.set_xticklabels(d.keys().tolist(), rotation=90)
    else:
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.show()


"""Visualization Utilities."""
import os
import random
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps

def plot_query_reference_samples(
    query_data_root: str,
    reference_data_root: str,
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    query_label_col_name: str,
    reference_label_col_name: str,
    query_img_col_name: str,
    reference_img_col_name: str,
    num_labels_to_sample: int = 5,
    num_samples_per_label: int = 5,
    subplot_width: int = 3,
):
    """Plot Query-Reference Samples.

    Args:
        query_data_root (str): Image root directory of query images.
        reference_data_root (str): Image root directory of reference images.
        query_df (pd.DataFrame): Query data frame.
        reference_df (pd.DataFrame): Reference data frame.
        query_label_col_name (str): Column name of desired labels set under query_df.
        reference_label_col_name (str): Column name of desired labels set under reference_df.
        query_img_col_name (str): Column name of image path, relative to query_data_root, under query_df.
        reference_img_col_name (str): Column name of image path, relative to reference_data_root, under reference_df.
        num_labels_to_sample (int, optional): Number of label ids to sampled and plotted. Defaults to 5.
        num_samples_per_label (int, optional): Number of reference samples per query to be plotted. Defaults to 5.
        subplot_width (int, optional): Width of each subplot in the figure. Defaults to 3.

    Raises:
        AssertionError: If query_lable_col_name is not present in query_df.
        AssertionError: If reference_label_col_name is not present in reference_df.
        AssertionError: If query_img_col_name is not present in query_df.
        AssertionError: If reference_df_img_col_name is not present in reference_df.
        AssertionError: If there are no reference samples available for sampled query.
    """
    # sanity checks
    if query_label_col_name not in query_df:
        raise AssertionError("query_lable_col_name '{}' is not present in query_df.".format(query_label_col_name))

    if reference_label_col_name not in reference_df:
        raise AssertionError(
            "reference_label_col_name '{}' is not present in reference_df.".format(query_label_col_name)
        )

    if query_img_col_name not in query_df:
        raise AssertionError("query_img_col_name '{}' is not present in query_df.".format(query_img_col_name))

    if reference_img_col_name not in reference_df:
        raise AssertionError(
            "reference_df_img_col_name '{}' is not present in reference_df.".format(query_img_col_name)
        )

    # sample label ids
    label_ids = query_df[query_label_col_name].unique().tolist()
    num_labels_to_sample = min(num_labels_to_sample, len(label_ids))
    sample_label_ids = random.sample(label_ids, num_labels_to_sample)

    # find min reference samples available per query
    num_reference_samples_per_query = []
    for sample_label_id in sample_label_ids:
        sample_df = (
            reference_df[reference_df[reference_label_col_name] == sample_label_id].copy().reset_index(drop=True)
        )
        if len(sample_df) == 0:
            raise AssertionError(
                "There are no reference samples available for query label '{}'".format(sample_label_id)
            )
        num_reference_samples_per_query.append(len(sample_df))
    num_samples_per_label = min(num_samples_per_label, min(num_reference_samples_per_query))

    # init subplots
    rows = num_labels_to_sample
    cols = num_samples_per_label + 1
    _, axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(cols * subplot_width, rows * subplot_width), facecolor=(1, 1, 1), sharey=True
    )

    for i, sample_label_id in enumerate(sample_label_ids):
        query_sample_df = query_df[query_df[query_label_col_name] == sample_label_id].copy().reset_index(drop=True)
        reference_sample_df = (
            reference_df[reference_df[reference_label_col_name] == sample_label_id].copy().reset_index(drop=True)
        )

        query_img_file = query_sample_df[query_img_col_name].sample(1).item()
        query_img_file = os.path.join(query_data_root, query_img_file)

        reference_img_files = reference_sample_df[reference_img_col_name].sample(num_samples_per_label).tolist()

        axs[i][0].set_title("Query ({})".format(sample_label_id))
        axs[i][0].set_xticks([])
        axs[i][0].set_yticks([])
        im = Image.open(query_img_file).convert("RGB")
        im.thumbnail((512,512))
        axs[i][0].imshow(im, aspect="auto")
        
        for j in range(cols - 1):
            reference_img_file = os.path.join(reference_data_root, reference_img_files[j])
            axs[i][j + 1].set_title("Reference {}".format(j + 1))
            axs[i][j + 1].set_xticks([])
            axs[i][j + 1].set_yticks([])
            im_2 = Image.open(reference_img_file)
            im_2.thumbnail((512,512))
            axs[i][j + 1].imshow(im_2, aspect="auto")

    plt.tight_layout()
