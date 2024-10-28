import os
import random
from dgl.data import DGLDataset
from graphs import create_one_graph
from dgl import save_graphs, load_graphs

class CGRDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：
    Parameters
    ----------
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self,
                 save_dir,
                 force_reload=False,
                 verbose=True):
        super(CGRDataset, self).__init__(name='CGRDataset',
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def process(self):
        # 将原始数据处理为图、标签和数据集划分的掩码
        root_file=os.getcwd()+"/data/"
        data_files_all = os.listdir(root_file)
        origin_graphs = []
        for i in range(len(data_files_all)):
            g = create_one_graph(root_file + "data" + str(i) + "/connect.xlsx",
                                 root_file + "data" + str(i) + "/x.xlsx",
                                 root_file + "data" + str(i) + "/y.xlsx",
                                 root_file + "data" + str(i) + "/joint_coordinate.xlsx",
                                 root_file + "data" + str(i) + "/connect_comp.xlsx")
            origin_graphs.append(g)
        random.shuffle(origin_graphs)
        self.graphs=origin_graphs

    def __getitem__(self, idx):
        # 通过idx得到与之对应的一个样本
        return self.graphs[idx]

    def __len__(self):
        # 数据样本的数量
        return len(self.graphs)

    def save(self):
        # 将处理后的数据保存至 `self.save_path`
        g_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(g_path,self.graphs)

    def load(self):
        # 从 `self.save_path` 导入处理后的数据
        g_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self.graphs, label_dict = load_graphs(g_path)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)


if __name__=="__main__":
    save_dir=os.getcwd()+"/dataset"
    dataset = CGRDataset(save_dir=save_dir)

