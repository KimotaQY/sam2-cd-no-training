import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv("test_output.csv")

# 查看数据结构
print(df.head())

# 查看唯一的iou阈值
iou_thresholds = sorted(df["iou_threshold"].unique())
print("可用的IoU阈值:", iou_thresholds)

# 查看唯一的mid_frame值
mid_frames = sorted(df["mid_frame"].unique())
print("可用的mid_frame值:", mid_frames)

model_param = {
    "t": "tiny",
    "s": "small",
    "b+": "base plus",
    "l": "large",
}


def plot_f1_by_iou_seaborn(mid_frame_value, save_path=None):
    """
    使用seaborn绘制更美观的图表

    参数:
        mid_frame_value: 要分析的mid_frame值
        save_path: 图片保存路径(可选)
    """
    # 筛选数据
    subset = df[df["mid_frame"] == mid_frame_value]

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.set_palette("husl")  # 使用漂亮的颜色方案

    # 创建折线图
    ax = sns.lineplot(
        data=subset,
        x="iou_threshold",
        y="f1",
        hue="Model_type",
        style="Model_type",
        markers=True,
        dashes=False,
        markersize=10,
        linewidth=2.5,
    )

    # plt.title(
    #     f"F1 Score vs IoU Threshold (mid_frame={mid_frame_value})", fontsize=16, pad=20
    # )
    plt.xlabel("IoU Threshold", fontsize=14, labelpad=10)
    plt.ylabel("F1 Score", fontsize=14, labelpad=10)

    # 调整图例
    plt.legend(
        title="Model Type",
        title_fontsize=12,
        fontsize=11,
        # loc="upper right" if mid_frame_value == 0 else "lower right",
        loc="upper right",
    )

    # 调整坐标轴
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_f1_by_midframe(
    iou_threshold,
    prompt_type,
    diff_frame_num,
    save_path=None,
    figsize=(6, 4),
    dpi=300,
    style="default",
):
    """
    绘制符合SCI期刊标准的折线图，展示不同model_type的f1随mid_frame变化

    参数:
        iou_threshold: 固定使用的iou阈值
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
        style: 样式类型 ("default" 或 "large")
    """
    # 筛选指定iou阈值的数据
    subset = df[
        (df["iou_threshold"] == iou_threshold)
        & (df["prompt_type"] == prompt_type)
        & (df["diff_frame_num"] == diff_frame_num)
    ].copy()

    # 设置不同样式的字体大小和符号等参数
    if style == "large":
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
        }
    else:
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.linewidth": 0.8,
        }

    # 设置SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            **{
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "savefig.bbox": "tight",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.4,
            },
            **font_settings,
        }
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 定义模型参数映射 (根据您的实际需求修改)
    model_param = {"t": "Tiny", "s": "Small", "b+": "Base+", "l": "Large"}

    # 颜色和标记样式 - 使用色盲友好的调色板
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]  # 圆形,方形,菱形,三角形

    # 确保mid_frame排序正确
    mid_frames = sorted(subset["mid_frame"].unique())

    # 绘制每条折线
    for i, model in enumerate(["t", "s", "b+", "l"]):
        model_data = subset[subset["model_type"] == model]
        if not model_data.empty:
            # 按mid_frame排序
            model_data = model_data.sort_values("mid_frame")
            ax.plot(
                model_data["mid_frame"],
                model_data["f1"],
                label=model_param.get(model, model),
                color=colors[i],
                marker=markers[i],
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=3,
            )

    # 设置坐标轴标签
    ax.set_xlabel("mid_frame", labelpad=5)
    ax.set_ylabel("F1 Score", labelpad=5)

    # 设置图例
    if style == "default":
        ax.legend(
            title="Model Type",
            frameon=True,
            edgecolor="black",
            facecolor="white",
            # bbox_to_anchor=(1.02, 1),  # 将图例放在图形外侧右侧
            loc="upper right",
        )

    # 设置网格线
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # 设置x轴为整数刻度（如果mid_frame是整数）
    if all(isinstance(x, (int, np.integer)) for x in mid_frames):
        ax.set_xticks(mid_frames)

    # 设置y轴范围
    y_min, y_max = subset["f1"].min(), subset["f1"].max()
    ax.set_ylim(max(0, y_min - 0.05), min(1, y_max + 0.05))

    # 调整边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 添加标题
    # ax.set_title(f"IoU Threshold = {iou_threshold}", pad=10, fontsize=11)

    # 紧凑布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # plt.show()
    plt.close()


def plot_f1_by_midframe_bar(iou_threshold, save_path=None):
    """
    使用柱状图比较不同mid_frame的表现

    参数:
        iou_threshold: 要分析的iou阈值
        save_path: 图片保存路径(可选)
    """
    subset = df[df["iou_threshold"] == iou_threshold]

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # 创建柱状图
    ax = sns.barplot(
        data=subset,
        x="mid_frame",
        y="f1",
        hue="Model_type",
        palette="husl",
        ci="sd",  # 显示标准差
        capsize=0.1,
    )

    plt.title(
        f"F1 Score Comparison Across mid_frame Values (IoU Threshold={iou_threshold})",
        fontsize=16,
        pad=20,
    )
    plt.xlabel("mid_frame", fontsize=14, labelpad=10)
    plt.ylabel("F1 Score", fontsize=14, labelpad=10)

    # 调整图例
    plt.legend(title="Model Type", title_fontsize=12, fontsize=11, loc="upper right")

    # 在柱子上方添加数值标签
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_f1_by_iou_threshold(
    mid_frame_value,
    prompt_type,
    diff_frame_num=1,
    save_path=None,
    figsize=(6, 4),
    dpi=300,
    style="default",
):
    """
    绘制符合SCI期刊标准的折线图，展示不同model_type的f1随iou阈值变化

    参数:
        mid_frame_value: 要分析的mid_frame值
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
        style: 样式类型 ("default" 或 "large")
    """
    # 筛选指定mid_frame的数据
    subset = df[
        (df["mid_frame"] == mid_frame_value)
        & (df["prompt_type"] == prompt_type)
        & (df["diff_frame_num"] == diff_frame_num)
    ].copy()

    # 设置不同样式的字体大小和符号等参数
    if style == "large":
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
        }
    else:
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.linewidth": 0.8,
        }

    # 设置SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            **{
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "savefig.bbox": "tight",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.4,
            },
            **font_settings,
        }
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 颜色和标记样式 - 使用色盲友好的调色板
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]  # 圆形,方形,菱形,三角形

    # 定义模型参数映射 (根据您的实际需求修改)
    model_param = {"t": "Tiny", "s": "Small", "b+": "Base+", "l": "Large"}

    # 绘制每条折线
    for i, model in enumerate(["t", "s", "b+", "l"]):
        model_data = subset[subset["model_type"] == model]
        if not model_data.empty:
            # 按iou_threshold排序
            model_data = model_data.sort_values("iou_threshold")
            ax.plot(
                model_data["iou_threshold"],
                model_data["f1"],
                label=model_param.get(model, model),
                color=colors[i],
                marker=markers[i],
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
                zorder=3,
            )

    # 设置坐标轴标签
    ax.set_xlabel("IoU Threshold", labelpad=5)
    ax.set_ylabel("F1 Score", labelpad=5)

    # 设置图例
    if style == "default":
        ax.legend(
            title="Model Param",
            frameon=True,
            edgecolor="black",
            facecolor="white",
            loc="upper right",
        )

    # 设置网格线
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # 设置坐标轴范围
    ax.set_xlim(left=0.25, right=0.75)  # 根据您的数据范围调整
    ax.set_ylim(bottom=0.5, top=0.9)  # 根据您的数据范围调整

    # 设置刻度
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))  # 每0.1一个主刻度
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))

    # 调整边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 紧凑布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # plt.show()
    plt.close()


def plot_sci_style_barchart(
    df,
    iou_threshold,
    mid_frame,
    diff_frame_num=1,
    save_path=None,
    figsize=(6, 4),
    dpi=300,
):
    """
    绘制符合SCI期刊标准的柱状图

    参数:
        iou_threshold: 目标iou阈值
        mid_frame: 目标mid_frame值
        save_path: 保存路径(如None则不保存)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    # 筛选数据
    subset = df[
        (df["iou_threshold"] == iou_threshold)
        & (df["mid_frame"] == mid_frame)
        & (df["diff_frame_num"] == diff_frame_num)
    ].copy()

    # 确保数据顺序一致
    model_order = ["tiny", "small", "base plus", "large"]  # 您可以根据需要调整顺序
    prompt_order = ["points", "box", "mask"]  # 您可以根据需要调整顺序
    subset["model_type"] = pd.Categorical(
        subset["model_type"], categories=model_order, ordered=True
    )
    subset["prompt_type"] = pd.Categorical(
        subset["prompt_type"], categories=prompt_order, ordered=True
    )
    subset.sort_values(["model_type", "prompt_type"], inplace=True)

    # 创建图形
    plt.figure(figsize=figsize, dpi=dpi)

    # 使用SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            "font.family": "Arial",  # 推荐使用Arial或Times New Roman
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,  # 坐标轴线宽
            "lines.linewidth": 1.5,  # 线宽
            "xtick.major.width": 0.8,  # x轴刻度线宽
            "ytick.major.width": 0.8,  # y轴刻度线宽
        }
    )

    # 设置颜色 - 使用色盲友好的调色板
    colors = ["#4e79a7", "#f28e2b", "#e15759"]  # 蓝,橙,红

    # 计算柱状图位置
    n_models = len(model_order)
    n_prompts = len(prompt_order)
    bar_width = 0.25
    spacing = 0.05
    x = np.arange(n_models)

    # 绘制柱状图
    bars = []
    for i, prompt in enumerate(prompt_order):
        prompt_data = subset[subset["prompt_type"] == prompt]
        heights = prompt_data["f1"].values
        x_pos = x + i * (bar_width + spacing)
        bar = plt.bar(
            x_pos,
            heights,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.7,
            label=prompt,
            zorder=3,
        )
        bars.append(bar)

        # 添加数据标签
        for pos, height in zip(x_pos, heights):
            plt.text(
                pos,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 设置坐标轴
    plt.xticks(x + (n_prompts - 1) * (bar_width + spacing) / 2, model_order)
    plt.xlabel("Model Param", labelpad=5)
    plt.ylabel("F1 Score", labelpad=5)

    # 设置y轴范围
    y_min, y_max = subset["f1"].min(), subset["f1"].max()
    plt.ylim(max(0, y_min - 0.05), min(1, y_max + 0.1))

    # 添加图例
    plt.legend(
        title="Prompt Type",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    # 添加网格线
    plt.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # 调整边框
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.8)

    # 标题
    # plt.title(
    #     f"IoU Threshold={iou_threshold}, mid_frame={mid_frame}", pad=10, fontsize=10
    # )

    # 调整布局
    plt.tight_layout()

    if save_path:
        # 支持多种格式保存
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi)

    plt.show()


def get_f1_vs_midframe_data(iou_threshold):
    """
    绘制符合SCI期刊标准的折线图，并可选择返回整理好的数据

    参数:
        iou_threshold: iou阈值

    返回:
        返回pd.DataFrame
    """
    # 筛选指定iou阈值的数据
    subset = df[df["iou_threshold"] == iou_threshold].copy()

    # 整理数据为宽格式（每个model_type一列）
    data_pivot = subset.pivot_table(
        index="mid_frame",
        columns="Model_type",
        values="f1",
        aggfunc="mean",  # 如果有重复数据则取平均
    ).reset_index()

    # 添加iou_threshold列
    data_pivot["iou_threshold"] = iou_threshold

    # 重新排列列顺序
    cols = ["iou_threshold", "mid_frame"] + [
        m for m in ["t", "s", "b+", "l"] if m in data_pivot.columns
    ]
    data_pivot = data_pivot[cols]

    return data_pivot


def get_f1_vs_iou_threshold_data(mid_frame):
    """
    绘制符合SCI期刊标准的折线图，并可选择返回整理好的数据

    参数:
        mid_frame: 中间帧数量

    返回:
        返回pd.DataFrame
    """
    # 筛选指定iou阈值的数据
    subset = df[df["mid_frame"] == mid_frame].copy()

    # 整理数据为宽格式（每个model_type一列）
    data_pivot = subset.pivot_table(
        index="iou_threshold",
        columns="Model_type",
        values="f1",
        aggfunc="mean",  # 如果有重复数据则取平均
    ).reset_index()

    # 添加mid_frame列
    data_pivot["mid_frame"] = mid_frame

    # 重新排列列顺序
    cols = ["iou_threshold", "mid_frame"] + [
        m for m in ["t", "s", "b+", "l"] if m in data_pivot.columns
    ]
    data_pivot = data_pivot[cols]

    return data_pivot


def plot_memory_by_obj_num(
    save_path=None, figsize=(6, 4), dpi=300, model_type=None, style="default"
):
    """
    绘制符合SCI期刊标准的折线图，展示不同 prompt_type、mid_frame 的 max_memory_usage 随 obj_num 变化

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
        model_type: 模型类型 (可选)
        style: 样式类型 ("default" 或 "large")
    """
    # 设置不同样式的字体大小和符号等参数
    if style == "large":
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "lines.linewidth": 2,
            "lines.markersize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
        }
    else:
        font_settings = {
            "font.family": "Arial",  # 或 'Times New Roman'
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.linewidth": 0.8,
        }

    # Step 1: Read the data from log.csv
    df = pd.read_csv("log.csv")
    print(df.head())

    # 获取所有数据行的索引
    all_indices = df.index.tolist()

    # 选择单数行（第1、3、5...行）→ 对应索引为偶数（因为从0开始）
    odd_rows = df.iloc[::2]  # 步长为2，从第0行开始

    # 选择双数行（第2、4、6...行）→ 对应索引为奇数
    even_rows = df.iloc[1::2]  # 步长为2，从第1行开始

    df = even_rows
    if model_type is not None:
        df = df[df["model_type"] == model_type]

    # 示例输出前几行
    # print("单数行数据:")
    # print(odd_rows.head())

    # print("\n双数行数据:")
    # print(even_rows.head())

    # print(df[df["prompt_type"] == "box"].head())

    # Step 2: Ensure numeric types for relevant columns
    df["obj_num"] = pd.to_numeric(df["obj_num"], errors="coerce")
    df["max_memory_usage"] = pd.to_numeric(df["max_memory_usage"], errors="coerce")

    # Drop any rows with missing values in key columns
    df.dropna(
        subset=["obj_num", "max_memory_usage", "prompt_type", "mid_frame"], inplace=True
    )

    print(df.to_string())

    # Step 3: 设置SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            **{
                "figure.dpi": dpi,
                "savefig.dpi": dpi,
                "savefig.bbox": "tight",
                "grid.linewidth": 0.6,
                "grid.alpha": 0.4,
            },
            **font_settings,
        }
    )

    # Step 4: 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Step 5: 定义颜色和标记样式 - 每个 mid_frame 一种颜色
    mid_frames = sorted(df["mid_frame"].unique())
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]  # 圆形,方形,菱形,三角形
    # 不同线型
    linestyles = ["-", "--", ":", "-."]

    # Step 6: 按照 mid_frame 分组绘制每条折线
    for i, mid_frame in enumerate(mid_frames):
        mid_frame_data = df[df["mid_frame"] == mid_frame]

        # 按 prompt_type 分组绘制每条折线
        # prompt_types = sorted(mid_frame_data["prompt_type"].unique())
        prompt_types = ["box", "mask"]
        for j, prompt in enumerate(prompt_types):
            prompt_data = mid_frame_data[mid_frame_data["prompt_type"] == prompt]
            if not prompt_data.empty:
                # 按 obj_num 排序
                prompt_data = prompt_data.sort_values("obj_num")
                ax.plot(
                    prompt_data["obj_num"],
                    prompt_data["max_memory_usage"],
                    label=f"{prompt}_{mid_frame}",
                    color=colors[i % len(colors)],
                    marker=markers[j % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.5,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=3,
                )
            else:
                print(
                    f"⚠️ No data found for mid_frame '{mid_frame}' and prompt_type '{prompt}'"
                )

    # Step 7: 设置坐标轴标签
    ax.set_xlabel("Object Number", labelpad=5)
    ax.set_ylabel("Max Memory Usage (MB)", labelpad=5)

    # Step 8: 设置图例
    # 图例放在左上

    ax.legend(
        # title="Prompt Type",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=8 if style == "default" else 11,
        ncol=1,
    )

    # Step 9: 设置网格线
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Step 10: 设置x轴为整数刻度（如果obj_num是整数）
    obj_nums = sorted(df["obj_num"].unique())
    if all(isinstance(x, (int, float)) and x.is_integer() for x in obj_nums):
        ax.set_xticks(obj_nums)

    # Step 11: 调整边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Step 12: 紧凑布局
    plt.tight_layout()

    # Step 13: 保存图像
    if save_path:
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_timecost_by_obj_num(save_path=None, figsize=(6, 4), dpi=300):
    """
    绘制符合SCI期刊标准的折线图，展示不同 prompt_type、mid_frame 的 time_cost 随 obj_num 变化

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    # Step 1: Read the data from log.csv
    df = pd.read_csv("log.csv")
    print(df.head())

    # 获取所有数据行的索引
    all_indices = df.index.tolist()

    # 选择单数行（第1、3、5...行）→ 对应索引为偶数（因为从0开始）
    odd_rows = df.iloc[::2]  # 步长为2，从第0行开始

    # 选择双数行（第2、4、6...行）→ 对应索引为奇数
    even_rows = df.iloc[1::2]  # 步长为2，从第1行开始

    # df = even_rows
    df = even_rows[even_rows["model_type"] == "t"]
    # df = df[df["prompt_type"] == "box"]

    # 示例输出前几行
    # print("单数行数据:")
    # print(odd_rows.head())

    # print("\n双数行数据:")
    # print(even_rows.head())

    print(df[df["prompt_type"] == "box"].head())

    # Step 2: Ensure numeric types for relevant columns
    df["obj_num"] = pd.to_numeric(df["obj_num"], errors="coerce")
    df["time_cost"] = pd.to_numeric(df["time_cost"], errors="coerce")

    # Drop any rows with missing values in key columns
    df.dropna(subset=["obj_num", "time_cost", "prompt_type", "mid_frame"], inplace=True)

    # Step 3: 设置SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.4,
        }
    )

    # Step 4: 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Step 5: 定义颜色和标记样式 - 每个 mid_frame 一种颜色
    mid_frames = sorted(df["mid_frame"].unique())
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]  # 圆形,方形,菱形,三角形
    linestyles = ["-", "--", ":", "-."]  # 不同线型

    # Step 6: 按照 mid_frame 分组绘制每条折线
    for i, mid_frame in enumerate(mid_frames):
        mid_frame_data = df[df["mid_frame"] == mid_frame]

        # 按 prompt_type 分组绘制每条折线
        # prompt_types = sorted(mid_frame_data["prompt_type"].unique())
        prompt_types = ["box", "mask"]
        for j, prompt in enumerate(prompt_types):
            prompt_data = mid_frame_data[mid_frame_data["prompt_type"] == prompt]
            if not prompt_data.empty:
                # 按 obj_num 排序
                prompt_data = prompt_data.sort_values("obj_num")
                ax.plot(
                    prompt_data["obj_num"],
                    prompt_data["time_cost"],
                    label=f"{prompt}_{mid_frame}",
                    color=colors[i % len(colors)],
                    marker=markers[j % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.5,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=3,
                )
            else:
                print(
                    f"⚠️ No data found for mid_frame '{mid_frame}' and prompt_type '{prompt}'"
                )

    # Step 7: 设置坐标轴标签
    ax.set_xlabel("Object Number", labelpad=5)
    ax.set_ylabel("Time Cost (s)", labelpad=5)

    # Step 8: 设置图例
    # 图例放在左上

    ax.legend(
        title="Prompt Type",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=8,
        ncol=1,
    )

    # Step 9: 设置网格线
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # Step 10: 设置x轴为整数刻度（如果obj_num是整数）
    obj_nums = sorted(df["obj_num"].unique())
    if all(isinstance(x, (int, float)) and x.is_integer() for x in obj_nums):
        ax.set_xticks(obj_nums)

    # Step 11: 调整边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # Step 12: 紧凑布局
    plt.tight_layout()

    # Step 13: 保存图像
    if save_path:
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_boxplot_iou_vs_f1(
    save_path=None,
    figsize=(10, 6),
    dpi=300,
    style="default",
    model_type=None,
    prompt_type=None,
):
    """
    绘制箱线图：x轴为IoU阈值，y轴为F1得分

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    plt.figure(figsize=figsize, dpi=dpi)

    subset = df.copy()
    if model_type is not None:
        subset = subset[subset["model_type"] == model_type]
    if prompt_type is not None:
        subset = subset[subset["prompt_type"] == prompt_type]

    # 设置不同样式的字体大小和符号等参数
    font_scale = 1.0 if style == "default" else 2.0

    font_settings = {
        "font.family": "Arial",
        "font.size": 10 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 10 * font_scale,
    }
    boxprops = {"linewidth": 1 * font_scale, "edgecolor": "black"}
    medianprops = {"linewidth": 1 * font_scale, "color": "red"}

    sns.set(style="whitegrid")
    plt.rcParams.update(font_settings)

    ax = sns.boxplot(
        data=subset,
        x="iou_threshold",
        y="f1",
        palette="Set2",
        boxprops=boxprops,
        medianprops=medianprops,
        width=0.5,
    )

    ax.set_xlabel("IoU Threshold", labelpad=10)
    ax.set_ylabel("F1 Score", labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_boxplot_model_type_vs_f1(
    save_path=None, figsize=(10, 6), dpi=300, style="default", diff_frame_num=None
):
    """
    绘制箱线图：x轴为模型参数名，y轴为F1得分

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # 映射简写模型类型到完整名称
    model_param = {"t": "Tiny", "s": "Small", "b+": "Base+", "l": "Large"}
    df["Model_type_full"] = df["model_type"].map(model_param)
    subset = df.copy()
    if diff_frame_num is not None:
        subset = df[(df["diff_frame_num"] == diff_frame_num)].copy()

    # 设置不同样式的字体大小和符号等参数
    font_scale = 1.0 if style == "default" else 2.0

    font_settings = {
        "font.family": "Arial",
        "font.size": 10 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 10 * font_scale,
    }
    boxprops = {"linewidth": 1 * font_scale, "edgecolor": "black"}
    medianprops = {"linewidth": 1 * font_scale, "color": "red"}
    sns.set(style="whitegrid")
    plt.rcParams.update(font_settings)

    ax = sns.boxplot(
        data=subset,
        x="Model_type_full",
        y="f1",
        palette="Set2",
        boxprops=boxprops,
        medianprops=medianprops,
        width=0.5,
    )

    ax.set_xlabel("Model Type", labelpad=10)
    ax.set_ylabel("F1 Score", labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_boxplot_mid_frame_vs_f1(
    save_path=None,
    figsize=(10, 6),
    dpi=300,
    style="default",
    diff_frame_num=None,
    model_type=None,
    prompt_type=None,
):
    """
    绘制箱线图：x轴为中间帧数量，y轴为F1得分

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # subset = df.copy()
    subset = df[(df["mid_frame"].isin([1, 2, 3]))].copy()
    if diff_frame_num is not None:
        subset = df[(df["diff_frame_num"] == diff_frame_num)].copy()
    if model_type is not None:
        subset = subset[subset["model_type"] == model_type]
    if prompt_type is not None:
        subset = subset[subset["prompt_type"] == prompt_type]

    # 设置不同样式的字体大小和符号等参数
    font_scale = 1.0 if style == "default" else 2.0

    font_settings = {
        "font.family": "Arial",
        "font.size": 10 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 10 * font_scale,
    }
    boxprops = {"linewidth": 1 * font_scale, "edgecolor": "black"}
    medianprops = {"linewidth": 1 * font_scale, "color": "red"}
    sns.set(style="whitegrid")
    plt.rcParams.update(font_settings)

    ax = sns.boxplot(
        data=subset,
        x="mid_frame",
        y="f1",
        palette="Set2",
        boxprops=boxprops,
        medianprops=medianprops,
        width=0.5,
    )

    ax.set_xlabel("Mid Frame", labelpad=10)
    ax.set_ylabel("F1 Score", labelpad=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


def plot_timecost_by_obj_num_new(save_path=None, figsize=(6, 4), dpi=300, show_ci=True):
    """
    绘制符合SCI期刊标准的折线图（增强版），展示不同 prompt_type、mid_frame 的 time_cost 随 obj_num 变化

    参数:
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
        show_ci: 是否显示置信区间
    """
    df = pd.read_csv("log.csv")
    all_indices = df.index.tolist()
    odd_rows = df.iloc[::2]
    even_rows = df.iloc[1::2]
    # df_data = even_rows[even_rows["model_type"] == "l"]
    df_data = even_rows

    # 数据预处理
    df_data["obj_num"] = pd.to_numeric(df_data["obj_num"], errors="coerce")
    df_data["time_cost"] = pd.to_numeric(df_data["time_cost"], errors="coerce")
    df_data.dropna(
        subset=["obj_num", "time_cost", "prompt_type", "mid_frame"], inplace=True
    )

    # 样式设置
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.bbox": "tight",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.4,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    mid_frames = sorted(df_data["mid_frame"].unique())
    prompt_types = ["box", "mask"]

    # 颜色和样式定义
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]
    linestyles = ["-", "--", ":", "-."]

    for i, mid_frame in enumerate(mid_frames):
        mid_data = df_data[df_data["mid_frame"] == mid_frame]

        for j, prompt in enumerate(prompt_types):
            prompt_data = mid_data[mid_data["prompt_type"] == prompt].copy()

            if not prompt_data.empty:
                prompt_data = prompt_data.sort_values("obj_num")

                # 计算95%置信区间
                if show_ci:
                    sns.lineplot(
                        data=prompt_data,
                        x="obj_num",
                        y="time_cost",
                        ci=95,
                        n_boot=1000,
                        color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)],
                        ax=ax,
                    )

                # 主折线
                ax.plot(
                    prompt_data["obj_num"],
                    prompt_data["time_cost"],
                    label=f"{prompt}_{mid_frame}",
                    color=colors[i % len(colors)],
                    marker=markers[j % len(markers)],
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=1.5,
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=3,
                )

    # 坐标轴设置
    ax.set_xlabel("Object Number", labelpad=5)
    ax.set_ylabel("Time Cost (s)", labelpad=5)

    # 图例设置
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        title="Prompt_MidFrame",
        frameon=True,
        edgecolor="black",
        facecolor="white",
        loc="upper left",
        bbox_to_anchor=(0, 1),
        fontsize=8,
        ncol=1,
        borderaxespad=0.0,
    )

    # 网格和布局
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    obj_nums = sorted(df_data["obj_num"].unique())
    if all(isinstance(x, (int, float)) and x.is_integer() for x in obj_nums):
        ax.set_xticks(obj_nums)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()

    if save_path:
        if save_path.endswith(".tif"):
            plt.savefig(
                save_path,
                format="tiff",
                dpi=dpi,
                pil_kwargs={"compression": "tiff_lzw"},
            )
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()


# output_dir = "output_plot/compare_diff_mid"
# os.makedirs(output_dir, exist_ok=True)

# 示例：绘制iou_threshold=0.x时的柱状图
# plot_f1_by_midframe_bar(iou_threshold=0.4)

# 示例：绘制iou_threshold=0.x时的曲线
# plot_f1_by_midframe(iou_threshold=0.5, save_path="f1_by_midframe_iou0.5.jpg")

# 示例：绘制mid_frame=x时的曲线
# plot_f1_by_iou_seaborn(mid_frame_value=1, save_path="f1_by_iou_midframe1.jpg")

# 示例：绘制mid_frame=x时的曲线
# plot_f1_by_iou_threshold(mid_frame_value=0, save_path="f1_by_iou_midframe0.jpg")

###### 绘制所有prompt_type的综合比较 ######
# df = pd.read_csv("test_output.csv")
# plot_sci_style_barchart(
#     df,
#     iou_threshold=0.5,
#     mid_frame=1,
#     save_path="sci_style_barchart.png",
#     figsize=(6, 4),
#     dpi=300,
# )

###### 绘制mid_frame=x时的曲线
# df = pd.read_csv("./charts/output.csv")
# df = pd.read_csv("test_output.csv")
# for mid_frame_value in [1, 2, 3]:
#     for prompt_type in ["box", "points", "mask"]:
#         for diff_frame_num in [1, -1]:
#             # mid_frame_value = 3
#             # prompt_type = "box"
#             # diff_frame_num = 1
#             save_path = os.path.join(
#                 output_dir,
#                 f"f1_by_iou_mid{mid_frame_value}_{prompt_type}_{diff_frame_num}.png",
#             )
#             plot_f1_by_iou_threshold(
#                 mid_frame_value=mid_frame_value,
#                 prompt_type=prompt_type,
#                 diff_frame_num=diff_frame_num,
#                 save_path=save_path,
#                 figsize=(6, 4),
#                 dpi=300,
#                 style="large",
#             )

###### 绘制iou=x时的曲线
# output_dir = "output_plot/compare_diff_mid"
# os.makedirs(output_dir, exist_ok=True)
# for iou_threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
#     for prompt_type in ["box", "points", "mask"]:
#         for diff_frame_num in [1, -1]:
#             save_path = os.path.join(
#                 output_dir,
#                 f"f1_by_midframe_iou{iou_threshold}_{prompt_type}_{diff_frame_num}.png",
#             )
#             plot_f1_by_midframe(
#                 iou_threshold=iou_threshold,
#                 prompt_type=prompt_type,
#                 diff_frame_num=diff_frame_num,
#                 save_path=save_path,
#                 figsize=(6, 4),
#                 dpi=300,
#                 style="large",
#             )

###### 绘制箱型图
# output_dir = "output_plot/boxplot"
# os.makedirs(output_dir, exist_ok=True)
# for prompt_type in ["points", "box", "mask"]:
#     # for model_type in ["t", "s", "b+", "l"]:
#     # plot_boxplot_iou_vs_f1(
#     #     save_path=os.path.join(output_dir, f"boxplot_iou_vs_f1_{prompt_type}.png"),
#     #     style="large",
#     #     prompt_type=prompt_type,
#     # )
#     plot_boxplot_mid_frame_vs_f1(
#         save_path=os.path.join(
#             output_dir, f"boxplot_mid_frame_vs_f1_{prompt_type}.png"
#         ),
#         style="large",
#         prompt_type=prompt_type,
#     )
# plot_boxplot_model_type_vs_f1(
#     diff_frame_num=-1,
#     save_path=os.path.join(output_dir, "boxplot_model_type_vs_f1_d-1.png"),
#     style="large",
# )
###### End ######

# # result_data = get_f1_vs_midframe_data(0.5)
# result_data = get_f1_vs_iou_threshold_data(0)
# # 查看整理好的数据
# print(result_data.head())

# # 将数据保存为CSV
# result_data.to_csv("f1_vs_iou_threshold_midframe0_data.csv", index=False)

# 绘制不同prompt_type、不同obj_num的显存占用对比
output_dir = "output_plot"
for model_type in ["t", "s", "b+", "l"]:
    save_path = os.path.join(
        output_dir, f"plot_memory_by_obj_num_{model_type}_large.png"
    )
    plot_memory_by_obj_num(save_path=save_path, model_type=model_type, style="large")

# # 绘制不同prompt_type、不同obj_num的时间对比
# save_path = os.path.join(output_dir, "plot_timecost_by_obj_num.png")
# plot_timecost_by_obj_num(save_path=save_path)


# 读取原始CSV文件
# df = pd.read_csv("log.csv")
# print(df.head())

# # 获取所有数据行的索引
# all_indices = df.index.tolist()

# # 选择单数行（第1、3、5...行）→ 对应索引为偶数（因为从0开始）
# odd_rows = df.iloc[::2]  # 步长为2，从第0行开始

# # 选择双数行（第2、4、6...行）→ 对应索引为奇数
# even_rows = df.iloc[1::2]  # 步长为2，从第1行开始

# # 将单数行保存为新的CSV文件
# odd_rows.to_csv("odd_rows.csv", index=False)

# # 将双数行保存为新的CSV文件
# even_rows.to_csv("even_rows.csv", index=False)

###### 计算平均值 ######
# # 读取CSV文件
# df = pd.read_csv("even_rows.csv")  # 替换为你的CSV文件路径

# # 指定要过滤的条件（可以根据需要修改）
# model_type_filter = "t"  # 示例 model_type
# mid_frame_filter = 1  # 示例 mid_frame
# prompt_type_filter = "box"  # 示例 prompt_type
# obj_num_filter = 100  # 示例 obj_num

# # 应用过滤条件
# filtered_data = df[
#     (df["model_type"] == model_type_filter)
#     & (df["mid_frame"] == mid_frame_filter)
#     & (df["prompt_type"] == prompt_type_filter)
#     & (df["obj_num"] == obj_num_filter)
#     & (df["time_cost"] < 2)
# ]
# print(filtered_data.head())

# # 计算 time_cost 的平均值
# average_time_cost = filtered_data["time_cost"].mean()

# # 输出结果
# print(f"符合条件的 time_cost 平均值为: {average_time_cost}")

###### End ######
