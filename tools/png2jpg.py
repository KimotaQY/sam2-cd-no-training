import os
from PIL import Image

def convert_png_to_jpg(input_dir, output_dir, quality=95):
    """
    将 input_dir 下的所有 PNG 图片转换为 JPG 并保存到 output_dir
    
    参数:
        input_dir (str): 输入文件夹路径（包含PNG图片）
        output_dir (str): 输出文件夹路径（保存JPG图片）
        quality (int): JPG 质量（1-100，默认95）
    """
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.png'):
            # 构造输入和输出路径
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
            
            # 打开PNG图片并转换为RGB模式（JPG不支持PNG的RGBA透明度）
            try:
                with Image.open(input_path) as img:
                    if img.mode in ('RGBA', 'LA'):
                        # 创建一个白色背景的RGB图像
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])  # 使用alpha通道作为mask
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 保存为JPG
                    img.save(output_path, 'JPEG', quality=quality)
                    print(f"转换成功: {filename} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"转换失败 {filename}: {str(e)}")

# 示例用法
if __name__ == "__main__":
    input_folder = "D:\QY\Datasets\sam2_test\WHU"  # 替换为你的PNG图片所在文件夹
    output_folder = "D:\QY\Datasets\sam2_test\WHU\output_jpg"   # 替换为你想保存JPG的文件夹
    
    convert_png_to_jpg(input_folder, output_folder, quality=90)