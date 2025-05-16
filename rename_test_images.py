import os
import re
import tkinter as tk
from tkinter import filedialog, simpledialog

# 使用Tkinter弹窗选择目录和输入前缀
root = tk.Tk()
root.withdraw()

# 选择目录
folder_path = filedialog.askdirectory(title='请选择需要重命名的图片文件夹')
if not folder_path:
    print('未选择文件夹，程序退出。')
    exit(0)

# 输入前缀
prefix = simpledialog.askstring('输入前缀', '请输入重命名的前缀（如stranger）：')
if not prefix:
    print('未输入前缀，程序退出。')
    exit(0)

# 获取现有文件名和计数
pattern = re.compile(rf'{re.escape(prefix)}_(\d+)\.(jpg|jpeg|png)$', re.IGNORECASE)
existing_numbers = []
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        existing_numbers.append(int(match.group(1)))

next_counter = max(existing_numbers) + 1 if existing_numbers else 1

# 只重命名不是前缀_编号的文件
for filename in os.listdir(folder_path):
    if pattern.match(filename):
        continue
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        new_filename = f"{prefix}_{next_counter}{ext}"
        new_path = os.path.join(folder_path, new_filename)
        while os.path.exists(new_path):
            next_counter += 1
            new_filename = f"{prefix}_{next_counter}{ext}"
            new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"已重命名: {filename} -> {new_filename}")
        next_counter += 1

print(f"重命名完成: {prefix}_编号到 {next_counter-1}") 