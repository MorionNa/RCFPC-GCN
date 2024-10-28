# RCFPC-GCN: Assessing progressive collapse regions of reinforced concrete frame structures using Graph Convolutional Networks (Engineering Structures)
[[Paper]](https://doi.org/10.1016/j.engstruct.2024.119076)

本仓库为《Assessing Progressive Collapse Regions of Reinforced Concrete Frame Structures Using Graph Convolutional Networks》文章的工作内容，该文章已发表在《Engineering Structures》期刊上。
# requirement
1. [python 3.11](https://www.python.org/downloads/release/python-3118/)
2. [Blender 2.92](https://download.blender.org/release/Blender2.92/)
3. Windows 10.
# 安装环境
1. 将blender2.92放置在本项目的根目录下;
2. 运行blender_dependency.bat;
3. 运行Setup.bat即可。
# 使用说明
1. 将结构信息储存在不同的excel文件中，并放置于input文件夹下；
2. 清空output文件夹；
3. 运行Run.bat；
4. 点击下图红框中的图标进行可视化。
![image](https://github.com/user-attachments/assets/5c4e25e0-c072-4c36-9b3e-697345cb694e)
# 训练
1. 将输入文件放置于data文件夹下；
2. 运行dataset.py，dataset文件夹下会自动生成一个包含.bin文件的新文件夹；
3. 运行train.py进行训练。
