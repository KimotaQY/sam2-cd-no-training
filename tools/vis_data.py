import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv("output.csv")

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


def plot_f1_by_midframe(iou_threshold, save_path=None, figsize=(6, 4), dpi=300):
    """
    绘制符合SCI期刊标准的折线图，展示不同model_type的f1随mid_frame变化

    参数:
        iou_threshold: 固定使用的iou阈值
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    # 筛选指定iou阈值的数据
    subset = df[df["iou_threshold"] == iou_threshold].copy()

    # 设置SCI期刊推荐的样式
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
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.4,
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
        model_data = subset[subset["Model_type"] == model]
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

    plt.show()


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


def plot_f1_by_iou_threshold(mid_frame_value, save_path=None, figsize=(6, 4), dpi=300):
    """
    绘制符合SCI期刊标准的折线图，展示不同model_type的f1随iou阈值变化

    参数:
        mid_frame_value: 要分析的mid_frame值
        save_path: 图片保存路径(可选)
        figsize: 图像尺寸(英寸)
        dpi: 分辨率
    """
    # 筛选指定mid_frame的数据
    subset = df[df["mid_frame"] == mid_frame_value].copy()

    # 设置SCI期刊推荐的样式
    plt.style.use("default")  # 重置为默认样式
    plt.rcParams.update(
        {
            "font.family": "Arial",  # 或 'Times New Roman'
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

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 颜色和标记样式 - 使用色盲友好的调色板
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#59a14f"]  # 蓝,橙,红,绿
    markers = ["o", "s", "D", "^"]  # 圆形,方形,菱形,三角形

    # 定义模型参数映射 (根据您的实际需求修改)
    model_param = {"t": "Tiny", "s": "Small", "b+": "Base+", "l": "Large"}

    # 绘制每条折线
    for i, model in enumerate(["t", "s", "b+", "l"]):
        model_data = subset[subset["Model_type"] == model]
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

    plt.show()


def plot_sci_style_barchart(
    iou_threshold, mid_frame, save_path=None, figsize=(6, 4), dpi=300
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
        (df["iou_threshold"] == iou_threshold) & (df["mid_frame"] == mid_frame)
    ].copy()

    # 确保数据顺序一致
    model_order = ["tiny", "small", "base plus", "large"]  # 您可以根据需要调整顺序
    prompt_order = ["points", "box", "mask"]  # 您可以根据需要调整顺序
    subset["Model_type"] = pd.Categorical(
        subset["Model_type"], categories=model_order, ordered=True
    )
    subset["prompt_type"] = pd.Categorical(
        subset["prompt_type"], categories=prompt_order, ordered=True
    )
    subset.sort_values(["Model_type", "prompt_type"], inplace=True)

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


# 示例：绘制iou_threshold=0.x时的柱状图
# plot_f1_by_midframe_bar(iou_threshold=0.4)

# 示例：绘制iou_threshold=0.x时的曲线
# plot_f1_by_midframe(iou_threshold=0.5, save_path="f1_by_midframe_iou0.5.jpg")

# 示例：绘制mid_frame=x时的曲线
# plot_f1_by_iou_seaborn(mid_frame_value=1, save_path="f1_by_iou_midframe1.jpg")

# 示例：绘制mid_frame=x时的曲线
# plot_f1_by_iou_threshold(mid_frame_value=0, save_path="f1_by_iou_midframe0.jpg")

# 绘制所有prompt_type的综合比较
# plot_sci_style_barchart(
#     iou_threshold=0.5,
#     mid_frame=1,
#     save_path="sci_style_barchart.png",
#     figsize=(6, 4),
#     dpi=300,
# )

# 绘制mid_frame=x时的曲线
df = pd.read_csv("./charts/output.csv")
# mid_frame_value = 0
# plot_f1_by_iou_threshold(
#     mid_frame_value=mid_frame_value,
#     save_path=f"f1_by_iou_midframe{mid_frame_value}.png",
#     figsize=(6, 4),
#     dpi=300,
# )

# iou_threshold = 0.5
# plot_f1_by_midframe(
#     iou_threshold=iou_threshold,
#     save_path=f"f1_by_midframe_iou{iou_threshold}.png",
#     figsize=(6, 4),
#     dpi=300,
# )


# result_data = get_f1_vs_midframe_data(0.5)
result_data = get_f1_vs_iou_threshold_data(0)
# 查看整理好的数据
print(result_data.head())

# 将数据保存为CSV
result_data.to_csv("f1_vs_iou_threshold_midframe0_data.csv", index=False)
