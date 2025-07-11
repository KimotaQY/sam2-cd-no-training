import subprocess
import os


# 测试不同参数下的结果
def test_params():
    # 替换为你的 conda.exe 实际路径（一般在 Anaconda/Scripts 下）
    conda_path = r"E:/miniconda3/Scripts/conda.exe"

    # 检查 conda.exe 是否存在
    if not os.path.exists(conda_path):
        raise FileNotFoundError(f"conda.exe 未找到！请检查路径: {conda_path}")

    # 另一个conda环境的Python路径（需要替换为实际路径）
    other_env_python = r"E:\miniconda3\envs\sam2-cd-no-training"
    other_env_name = "sam2-cd-no-training"
    # 要执行的Python文件所在目录和模块名
    script_dir = r"E:\CD_projects\sam2-cd-no-training"
    script_path = r"E:\CD_projects\sam2-cd-no-training\\script_eval.py"
    module_name = "script_eval"  # 文件名（不带.py）

    iou_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    mid_list = [0, 1, 2, 3]
    diff_frame_num_list = [1, -1]
    # model_obj = ["t", "s", "b+", "l"]
    model_obj = ["l"]
    prompt_type_list = ["points", "box", "mask"]

    for model_type in model_obj:
        for diff_frame_num in diff_frame_num_list:
            for mid_frame in mid_list:
                for iou in iou_list:
                    for prompt_type in prompt_type_list:
                        # 要调用的函数名和参数
                        function_name = "main"
                        args = [
                            model_type,
                            mid_frame,
                            diff_frame_num,
                            iou,
                            prompt_type,
                        ]

                        # 使用 conda run 执行目标环境的 Python
                        cmd = [
                            conda_path,
                            "run",
                            "-n",
                            other_env_name,
                            "python",
                            "-c",
                            f"import sys; sys.path.append(r'{script_dir}'); from {module_name} import {function_name}; {function_name}(*{args})",
                        ]

                        result = subprocess.run(cmd, capture_output=True, text=True)
                        print(result.stdout)


# 测试不同模型生成的prompt效果
def test_prompt():
    # 替换为你的 conda.exe 实际路径（一般在 Anaconda/Scripts 下）
    conda_path = r"E:/miniconda3/Scripts/conda.exe"

    # 检查 conda.exe 是否存在
    if not os.path.exists(conda_path):
        raise FileNotFoundError(f"conda.exe 未找到！请检查路径: {conda_path}")

    # 另一个conda环境的Python路径（需要替换为实际路径）
    other_env_python = r"E:\miniconda3\envs\sam2-cd-no-training"
    other_env_name = "sam2-cd-no-training"
    # 要执行的Python文件所在目录和模块名
    script_dir = r"E:\CD_projects\sam2-cd-no-training"
    script_path = r"E:\CD_projects\sam2-cd-no-training\\script_eval.py"
    module_name = "script_eval"  # 文件名（不带.py）

    dataset_names = ["WHU-CD"]
    for dataset_name in dataset_names:

        test_path = f"E:/CD_datasets/{dataset_name}/test/owlv2"
        # 获取test_path下的所有子目录，包含owlv2关键词
        test_dirs = [
            dir.split("_", 1)[1] for dir in os.listdir(test_path) if "owlv2" in dir
        ]

        iou_list = [0.5]
        mid_list = [1]
        diff_frame_num_list = [-1]
        model_obj = ["t", "s", "b+"]

        for model_type in model_obj:
            for diff_frame_num in diff_frame_num_list:
                for mid_frame in mid_list:
                    for iou in iou_list:
                        for test_dir in test_dirs:
                            T1 = f"E:/CD_datasets/{dataset_name}/test/A"
                            T2 = f"E:/CD_datasets/{dataset_name}/test/B"
                            diff_label_dir = f"E:/CD_datasets/{dataset_name}/test/label"
                            T1_label = f"E:/CD_datasets/{dataset_name}/test/owlv2/A_{test_dir}/result"
                            T2_label = f"E:/CD_datasets/{dataset_name}/test/owlv2/B_{test_dir}/result"

                            # 要调用的函数名和参数
                            function_name = "test"
                            args = [
                                model_type,
                                mid_frame,
                                diff_frame_num,
                                iou,
                                T1,
                                T2,
                                diff_label_dir,
                                T1_label,
                                T2_label,
                                dataset_name,
                            ]

                            # 使用 conda run 执行目标环境的 Python
                            cmd = [
                                conda_path,
                                "run",
                                "-n",
                                other_env_name,
                                "python",
                                "-c",
                                f"import sys; sys.path.append(r'{script_dir}'); from {module_name} import {function_name}; {function_name}(*{args})",
                            ]

                            result = subprocess.run(cmd, capture_output=True, text=True)
                            print(result.stdout)


# 测试不同参数的时间和内存开销
def test_time_memory_cost():
    # 替换为你的 conda.exe 实际路径（一般在 Anaconda/Scripts 下）
    conda_path = r"E:/miniconda3/Scripts/conda.exe"

    # 检查 conda.exe 是否存在
    if not os.path.exists(conda_path):
        raise FileNotFoundError(f"conda.exe 未找到！请检查路径: {conda_path}")

    # 另一个conda环境的Python路径（需要替换为实际路径）
    other_env_name = "sam2-cd-no-training"
    # 要执行的Python文件所在目录和模块名
    script_dir = r"E:\CD_projects\sam2-cd-no-training"
    module_name = "test_6"  # 文件名（不带.py）

    iou_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    mid_list = [0, 1, 2, 3]
    diff_frame_num_list = [1, -1]
    model_types = ["t", "s", "b+", "l"]
    prompt_types = ["points", "box", "mask"]
    obj_nums = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    count = 0
    for model_type in model_types:
        for diff_frame_num in [-1]:
            for mid_frame in [1]:
                for iou in [0.5]:
                    for prompt_type in ["box"]:
                        for obj_num in [1, 100]:
                            for _ in range(9):
                                # 要调用的函数名和参数
                                function_name = "main"
                                args = [
                                    model_type,
                                    mid_frame,
                                    diff_frame_num,
                                    iou,
                                    prompt_type,
                                    "whu",
                                    obj_num,
                                ]

                                # 使用 conda run 执行目标环境的 Python
                                cmd = [
                                    conda_path,
                                    "run",
                                    "-n",
                                    other_env_name,
                                    "python",
                                    "-c",
                                    f"import sys; sys.path.append(r'{script_dir}'); from {module_name} import {function_name}; {function_name}(*{args})",
                                ]

                                result = subprocess.run(
                                    cmd, capture_output=True, text=True
                                )
                                print(result.stdout)

                                count += 1
                                print(f"已执行第 {count} / 72 个任务")


if __name__ == "__main__":
    # test_params()
    # test_prompt()
    test_time_memory_cost()
