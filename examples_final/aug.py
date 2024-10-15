import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from itertools import repeat, product
import numpy as np

from copy import deepcopy
import pdb


class TUDataset_aug(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <https://chrsmrrs.github.io/datasets>`_.
    In addition, this dataset wrapper provides `cleaned dataset versions
    <https://github.com/nd7141/graph_datasets>`_ as motivated by the
    `"Understanding Isomorphism Bias in Graph Data Sets"
    <https://arxiv.org/abs/1910.12091>`_ paper, containing only non-isomorphic
    graphs.

    .. note::
        Some datasets may not come with any node labels.
        You can then either make use of the argument :obj:`use_node_attr`
        to load additional continuous node attributes (if present) or provide
        synthetic node features using transforms such as
        like :class:`torch_geometric.transforms.Constant` or
        :class:`torch_geometric.transforms.OneHotDegree`.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node attributes (if present).
            (default: :obj:`False`)
        use_edge_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous edge attributes (if present).
            (default: :obj:`False`)
        cleaned: (bool, optional): If :obj:`True`, the dataset will
            contain only non-isomorphic graphs. (default: :obj:`False`)
    """

    url = ('http://ls11-www.cs.tu-dortmund.de/people/morris/'
           'graphkerneldatasets')
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None, use_node_attr=False, use_edge_attr=False,
                 cleaned=False, aug=None,eva = None):
        self.name = name
        self.cleaned = cleaned
        self.eva = eva
        super(TUDataset_aug, self).__init__(root, transform, pre_transform,
                                        pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
        if not (self.name == 'MUTAG' or self.name == 'PTC_MR' or self.name == 'DD' or self.name == 'PROTEINS' or self.name == 'NCI1' or self.name == 'NCI109'):
            edge_index = self.data.edge_index[0, :].numpy()
            _, num_edge = self.data.edge_index.size()
            nlist = [edge_index[n] + 1 for n in range(num_edge - 1) if edge_index[n] > edge_index[n + 1]]
            nlist.append(edge_index[-1] + 1)

            num_node = np.array(nlist).sum()
            self.data.x = torch.ones((num_node, 1))

            edge_slice = [0]
            k = 0
            for n in nlist:
                k = k + n
                edge_slice.append(k)
            self.slices['x'] = torch.tensor(edge_slice)

            '''
            print(self.data.x.size())
            print(self.slices['x'])
            print(self.slices['x'].size())
            assert False
            '''

        self.aug = aug

    @property
    def raw_dir(self):
        name = 'raw{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed{}'.format('_cleaned' if self.cleaned else '')
        return osp.join(self.root, self.name, name)

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self):
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self):
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self):
        names = ['A', 'graph_indicator']
        return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        #print("root",self.root)
        #print("name", self.name)
        #print(self.raw_dir)
        #print("folder", osp.join(self.root, self.name))
        #url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        #path = download_url('{}/{}.zip'.format(url, self.name), folder)
        path =f"{self.name}.zip"
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)
    def process(self):
        self.data, self.slices, _ = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get_num_feature(self):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[0]

        for key in self.data.keys():
            try:
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key,
                                            item)] = slice(slices[0],
                                                           slices[0 + 1])
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
            except:
                pass
        _, num_feature = data.x.size()

        return num_feature


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            try:
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key,
                                            item)] = slice(slices[idx],
                                                           slices[idx + 1])
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]
            except:
                pass

        node_num = data.edge_index.max()
        sl = torch.tensor([[n,n] for n in range(node_num)]).t()
        data.edge_index = torch.cat((data.edge_index, sl), dim=1)

        if self.aug == 'minmax':
            n = np.random.choice(5, 1, p=self.aug_P)[0]
            data_aug = deepcopy(data)
            if n == 0:
                data_aug.edge_index, data_aug.x = custom1(data_aug.edge_index, data_aug.x)
                if isinstance(data_aug.edge_index, np.ndarray):
                    data_aug.edge_index = torch.from_numpy(data_aug.edge_index)
                if isinstance(data_aug.x, np.ndarray):
                    data_aug.x = torch.from_numpy(data_aug.x)
            elif n == 1:
                data_aug.edge_index, data_aug.x = custom1(data_aug.edge_index, data_aug.x)
                if isinstance(data_aug.edge_index, np.ndarray):
                    data_aug.edge_index = torch.from_numpy(data_aug.edge_index)
                if isinstance(data_aug.x, np.ndarray):
                    data_aug.x = torch.from_numpy(data_aug.x)
            elif n == 2:
                data_aug.edge_index, data_aug.x = custom3(data_aug.edge_index, data_aug.x)
                if isinstance(data_aug.edge_index, np.ndarray):
                    data_aug.edge_index = torch.from_numpy(data_aug.edge_index)
                if isinstance(data_aug.x, np.ndarray):
                    data_aug.x = torch.from_numpy(data_aug.x)
            elif n == 3:
                data_aug.edge_index, data_aug.x = custom4(data_aug.edge_index, data_aug.x)
                if isinstance(data_aug.edge_index, np.ndarray):
                    data_aug.edge_index = torch.from_numpy(data_aug.edge_index)
                if isinstance(data_aug.x, np.ndarray):
                    data_aug.x = torch.from_numpy(data_aug.x)
            else:
                pass
            # elif n == 5:
            #     data_aug = drop_nodes(deepcopy(data))
            # elif n == 6:
            #     data_aug.edge_index, data_aug.x = custom5(data_aug.edge_index, data_aug.x)
            # elif n == 7:
            #     data_aug = mask_nodes(deepcopy(data))
            # elif n == 8:
            #     data_aug = permute_edges(deepcopy(data))
            # elif n == 9:
            #     data_aug = subgraph(deepcopy(data))
        elif self.aug == "eva": # 使用传入的方法
            data_aug = deepcopy(data)
            data_aug.edge_index, data_aug.x = self.eva(data_aug.edge_index, data_aug.x)
        elif self.aug == "basic":
            data_aug = deepcopy(data)
            data_aug.edge_index, data_aug.x = self.eva(data_aug.edge_index, data_aug.x)
        elif self.aug == "none":
            data_aug = deepcopy(data)
        else:
            print('augmentation error')
            assert False

        return data, data_aug

def custom1(edge_index, x):
    edge_num = edge_index.size(1)
    mask_num = int(edge_num * 0.2)

    idx_mask = np.random.choice(edge_num, mask_num, replace=False)

    edge_index = edge_index[:, [i for i in range(edge_num) if i not in idx_mask]]

    return edge_index,x

def custom2(edge_index, x):
    edge_index = np.array(edge_index)
    x = np.array(x)

    num_nodes = x.shape[0]
    add_num = int(num_nodes * 0.1)

    x_nodes_to_add = np.random.choice(num_nodes, add_num, replace=False)
    constant_value = np.random.uniform(-1, 1)  # random constant value between -1 and 1

    x[x_nodes_to_add] += constant_value

    return edge_index, x


# def custom3(edge_index, x):
#     edge_index = np.array(edge_index)
#     x = np.array(x)
#
#     num_edges = edge_index.shape[1]
#     noise_num = int(num_edges * 0.1)
#
#     edges_to_add_noise = np.random.choice(num_edges, noise_num, replace=False)
#
#     noise = np.random.normal(loc=0, scale=0.05, size=(noise_num, x.shape[1]))
#     x[edge_index[0, edges_to_add_noise]] *= noise
#
#     return edge_index, x
#
def custom3(edge_index, x):
    num_nodes, _ = x.size()

    # Randomly select two nodes to create a new edge
    new_edge = torch.tensor([[np.random.randint(num_nodes), np.random.randint(num_nodes)]], dtype=torch.long).T

    # Insert the new edge into the edge_index
    edge_index = torch.cat([edge_index, new_edge], dim=1)

    return edge_index, x
#
def custom4(edge_index, x):
    edge_index = np.array(edge_index)
    x = np.array(x)

    num_nodes = x.shape[0]
    mask_num = int(num_nodes * 0.1)

    x_nodes_to_mask = np.random.choice(num_nodes, mask_num, replace=False)

    mask = np.random.choice([-1, 1], size=(mask_num, x.shape[1]))
    x[x_nodes_to_mask] *= mask

    return edge_index, x

# def custom1(edge_index, x):
#     node_num, _ = x.size()
#     mask_num = int(node_num * 0.3)
#
#     idx_mask = np.random.choice(node_num, mask_num, replace=False)
#     x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, x.size(1))), dtype=torch.float32)
#
#     edge_index_np = edge_index.numpy()
#     idx_drop = np.isin(edge_index_np, idx_mask)
#     idx_drop = np.any(idx_drop, axis=0)
#     edge_index_np = edge_index_np[:, ~idx_drop]
#
#     return edge_index,x

# def custom2(edge_index, x):
#     node_perm = torch.randperm(int(torch.max(edge_index)) + 1)
#     edge_index[0] = node_perm[edge_index[0]]
#     edge_index[1] = node_perm[edge_index[1]]
#
#     return edge_index, x
#
# def custom3(edge_index, x):
#     edge_num = edge_index.size(1)
#     mask_num = int(edge_num * 0.2)
#
#     idx_mask = np.random.choice(edge_num, mask_num, replace=False)
#     mask_edge = torch.tensor(idx_mask)
#
#     mask = torch.zeros(edge_num, dtype=torch.bool)
#     mask[mask_edge] = True
#
#     edge_index = edge_index[:, ~mask]
#
#     x = x[~torch.unique(edge_index.flatten())]
#
#     return edge_index, x
#
# def custom4(edge_index, x):
#     edge_num = edge_index.size(1)
#     mask_num = int(edge_num * 0.2)
#
#     idx_mask = np.random.choice(edge_num, mask_num, replace=False)
#
#     mask_edge = torch.ones(edge_num, dtype=torch.bool)
#     mask_edge[idx_mask] = False
#
#     edge_index = edge_index[:, mask_edge]
#
#     unique_nodes = torch.unique(edge_index)
#
#     new_node_map = torch.zeros(torch.max(edge_index) + 1, dtype=int)
#     for i in range(len(unique_nodes)):
#         new_node_map[unique_nodes[i]] = i
#
#     x = x[new_node_map]
#
#     return edge_index, x

def custom1(edge_index, x):
    node_num, feat_dim = x.size()
    mask_num = int(node_num * 0.5)  # Change the percentage of nodes to keep if desired

    idx_keep = np.random.choice(node_num, mask_num, replace=False)

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj[idx_keep][:, idx_keep]

    edge_index = adj.nonzero().t()

    return edge_index,x

def custom2(edge_index, x):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num * 0.2)

    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}

    edge_index = edge_index.numpy()

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    return edge_index,x

def custom3(edge_index, x, drop_prob=0.3):
    node_num, _ = x.size()
    drop_num = int(node_num * drop_prob)

    idx_drop = torch.randperm(node_num)[:drop_num]
    idx_nondrop = torch.tensor([n for n in range(node_num) if n not in idx_drop])

    mask = torch.ones(node_num)
    mask[idx_drop] = 0

    edge_index = edge_index[:, (mask[edge_index[0]] + mask[edge_index[1]]).type(torch.bool)]

    return edge_index,x

def custom4(edge_index, x, add_prob=0.4):
    edge_num = edge_index.size(1)
    add_num = int(edge_num * add_prob)

    additional_edges = torch.randint(0, x.size(0), (2, add_num))
    edge_index = torch.cat([edge_index, additional_edges], dim=1)

    return edge_index,x

# def drop_nodes(data):
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     drop_num = int(node_num * 0.2)
#
#     idx_drop = np.random.choice(node_num, drop_num, replace=False)
#     idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
#     idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}
#
#     edge_index = data.edge_index.numpy()
#
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj[idx_drop, :] = 0
#     adj[:, idx_drop] = 0
#     edge_index = adj.nonzero().t()
#
#     data.edge_index = edge_index
#     return data
#
# def permute_edges(data):
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     permute_num = int(edge_num * 0.2)
#
#     edge_index = data.edge_index.transpose(0, 1).numpy()
#
#     idx_add = np.random.choice(node_num, (permute_num, 2))
#
#     edge_index = edge_index[np.random.choice(edge_num, edge_num-permute_num, replace=False)]
#     data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
#     return data
#
# def subgraph(data):
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     sub_num = int(node_num * (1-0.2))
#
#     edge_index = data.edge_index.numpy()
#
#     idx_sub = [np.random.randint(node_num, size=1)[0]]
#     idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])
#
#     count = 0
#     while len(idx_sub) <= sub_num:
#         count = count + 1
#         if count > node_num:
#             break
#         if len(idx_neigh) == 0:
#             break
#         sample_node = np.random.choice(list(idx_neigh))
#         if sample_node in idx_sub:
#             continue
#         idx_sub.append(sample_node)
#         idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))
#
#     idx_drop = [n for n in range(node_num) if not n in idx_sub]
#     idx_nondrop = idx_sub
#     idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
#
#     edge_index = data.edge_index.numpy()
#
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj[idx_drop, :] = 0
#     adj[:, idx_drop] = 0
#     edge_index = adj.nonzero().t()
#
#     data.edge_index = edge_index
#     return data
#
# def mask_nodes(data):
#     node_num, feat_dim = data.x.size()
#     mask_num = int(node_num * 0.2)
#
#     idx_mask = np.random.choice(node_num, mask_num, replace=False)
#     data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)
#
#     return data

# def custom5(edge_index, x):
#     node_num, feat_dim = x.size()
#     mask_num = int(node_num * 0.3)  # change the mask percentage to 30%
#
#     idx_drop = np.random.choice(node_num, mask_num, replace=False)
#     idx_keep = np.setdiff1d(np.arange(node_num), idx_drop)
#
#     x = x[idx_keep]
#
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj[idx_drop] = 0
#     adj[:, idx_drop] = 0
#     adj = adj[idx_keep][:, idx_keep]
#
#     edge_index = adj.nonzero().t()
#
#     return edge_index,x