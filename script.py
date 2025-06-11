import subprocess


if __name__ == "__main__":
    # 替换为你的 conda.exe 实际路径（一般在 Anaconda/Scripts 下）
    conda_path = r"E:/miniconda3/Scripts/conda.exe"

    # 检查 conda.exe 是否存在
    import os

    if not os.path.exists(conda_path):
        raise FileNotFoundError(f"conda.exe 未找到！请检查路径: {conda_path}")

    # 另一个conda环境的Python路径（需要替换为实际路径）
    other_env_python = r"E:\miniconda3\envs\sam2-cd-no-training"
    other_env_name = "sam2-cd-no-training"
    # 要执行的Python文件所在目录和模块名
    script_dir = r"E:\CD_projects\sam2-cd-no-training"
    script_path = r"E:\CD_projects\sam2-cd-no-training\\test_4.py"
    module_name = "test_4"  # 文件名（不带.py）

    iou_list = [0.3, 0.4, 0.5, 0.6, 0.7]
    mid_list = [0, 1, 2, 3]
    diff_frame_num_list = [1, -1]
    model_obj = ["t", "s", "b+", "l"]
    prompt_type_list = ["points", "box", "mask"]

    for model_type in model_obj:
        for diff_frame_num in [1]:
            for mid_frame in [1]:
                for iou in [0.5]:
                    for prompt_type in prompt_type_list:
                        # 要调用的函数名和参数
                        function_name = "main"
                        args = [model_type, mid_frame, diff_frame_num, iou, prompt_type]

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
