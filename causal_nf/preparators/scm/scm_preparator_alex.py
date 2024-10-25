import networkx as nx
import numpy as np
import torch
import torch.utils.data as tdata

from causal_nf.datasets.scm_dataset import SCMDataset
from causal_nf.distributions.scm import SCM
# from causal_nf.preparators.base_preparator import IdentityScaler
from causal_nf.preparators.scm._base_distributions import pu_dict
from causal_nf.preparators.tabular_preparator import TabularPreparator
from causal_nf.sem_equations import sem_dict
from causal_nf.transforms import ConfoundedCausalEquations
from causal_nf.utils.graph import ancestor_matrix
from causal_nf.utils.io import dict_to_cn
from causal_nf.utils.scalers import StandardTransform#, StandardScaler


class SCMPreparator(TabularPreparator):
    def __init__(
        self,
        name,
        num_samples,
        hidden_vars: list,
        num_hidden,
        sem_name,
        base_version,
        base_distribution_name,
        correlations,
        means,
        variances,
        bernoulli_coef=None,
        uniform_a=None,
        uniform_b=None,
        laplace_diversity=None,
        type="torch",
        use_edge_attr=False,
        multiple_distributions=None,
        device="auto",
        **kwargs,
    ):

        self._num_samples = num_samples
        self.hidden_vars = hidden_vars # list of hidden variables
        self.num_hidden = len(hidden_vars) # new !!!
        self.sem_name = sem_name
        self.dataset = None
        self.use_edge_attr = use_edge_attr
        self.sem_fn = sem_dict[name](sem_name=sem_name)
        self.base_distribution_name = base_distribution_name
        self.correlations = correlations
        self.means = means
        self.variances = variances
        self.bernoulli_coef = bernoulli_coef
        self.uniform_a = uniform_a
        self.uniform_b = uniform_b
        self.laplace_diversity = laplace_diversity
        self.multiple_distributions = multiple_distributions
        self.type = type

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.complete_adjacency = self.sem_fn.adjacency  # It is a function! we will have to call it to get the adjacency matrix
        self.adjacency = self.get_severed_adjacency  # also a function, but returns the severed adjacency matrix

        sem = ConfoundedCausalEquations(
            functions=self.sem_fn.functions, inverses=self.sem_fn.inverses, derivatives=None, hidden_vars=self.hidden_vars
        )
        self.num_nodes_complete = len(self.sem_fn.functions) # number of nodes in the complete graph
        self.num_nodes = len(self.sem_fn.functions) - self.num_hidden # number of nodes in the severed graph
        # need to provide the complete list of nodes
        self.base_distribution = pu_dict[self.num_nodes_complete](
            base_distribution_name=base_distribution_name, 
            correlations=correlations, 
            means = means,
            variances = variances,
            base_version=base_version, 
            no_nodes=self.num_nodes_complete, 
            bernoulli_coef = bernoulli_coef,
            uniform_a = uniform_a,
            uniform_b = uniform_b,
            laplace_diversity = laplace_diversity,
            multiple_distributions = multiple_distributions,
            **kwargs
        )

        self.base_version = base_version
        self.scm = SCM(base_distribution=self.base_distribution, transform=sem)

        self.intervention_index_list = self.sem_fn.intervention_index_list()
        # update it to remove the hidden variables
        self.intervention_index_list = self.prune_intervention_index_list()

        super().__init__(name=name,
                         task="modeling",
                         device = self.device,
                         **kwargs)

    @classmethod
    def params(cls, dataset):
        if isinstance(dataset, dict):
            dataset = dict_to_cn(dataset)

        my_dict = {
            "name": dataset.name,
            "num_samples": dataset.num_samples,
            "num_hidden": dataset.num_hidden,
            "hidden_vars": dataset.hidden_vars,
            "sem_name": dataset.sem_name,
            "base_version": dataset.base_version,
            "use_edge_attr": dataset.use_edge_attr,
            "base_distribution_name": dataset.base_distribution_name,
            "correlations": dataset.correlations,   
            "means": dataset.means, 
            "bernoulli_coef" : dataset.bernoulli_coef,  
            "uniform_a" : dataset.uniform_a,    
            "uniform_b" : dataset.uniform_b,    
            "laplace_diversity" : dataset.laplace_diversity,    
            "variances": dataset.variances, 
            "multiple_distributions" : dataset.multiple_distributions,  
            "device": "cpu" # dataset.device
        }

        my_dict.update(TabularPreparator.params(dataset))

        return my_dict

    @classmethod
    def loader(cls, dataset):
        my_dict = SCMPreparator.params(dataset)

        return cls(**my_dict)

    def _x_dim(self):
        """ Needed to load the CNF model
        """
        return self.num_nodes

    def _c_dim(self):
        return self.num_hidden

    def get_severed_adjacency(self,add_diag=False):
        """ Remove the columns from the hidden variables
        """
        complete_adjacency_matrix = self. get_complete_adjacency(add_diag)
        adj_matrix_list = [i for i in range(len(complete_adjacency_matrix))]
        remaining_variables = [i for i in adj_matrix_list if i not in self.hidden_vars]
        # remaining_variables = list(set(adj_matrix_list) - set(self.hidden_vars))
        return self.complete_adjacency(add_diag)[remaining_variables][:, remaining_variables]
        # return self.complete_adjacency(add_diag)[self.num_hidden:, self.num_hidden:]

    def get_complete_adjacency(self, add_diag=False):
        return self.complete_adjacency(add_diag)

    def prune_intervention_index_list(self):
        """ self.intervention_index_list contains the indices of the variables that can be intervened upon,
        defined in each SEM - eg. [0, 1, 2]. But we don't observe the hidden variables, so we need to remove them 
        from the list, as we cannot intervene on them
        """
        # we cannot intervene in hidden variables
        # however, we rest 2 because we start counting from 0, since in intervene we add the number of hidden variables
        # in the model, we do not observe the hidden variables, so we should start by 0.
        return [i for i in self.intervention_index_list if i not in self.hidden_vars]
        # return [i for i in self.intervention_index_list if i >= self.num_hidden]

    def edge_attr_dim(self):
        if self.dataset.use_edge_attr:
            return self.dataset.edge_attr.shape[-1]
        else:
            return None

    def feature_names(self, latex=False):

        x_dim = self.x_dim()

        if latex:
            x_names = [f"$x_{{{i + 1 + self.num_hidden}}}$" for i in range(x_dim)]
            z_names = [f"$z_{{{i + 1}}}$" for i in range(self.num_hidden)]
            return z_names + x_names
        else:
            x_names = [f"x_{i + 1 + self.num_hidden}" for i in range(x_dim)]
            z_names = [f"z_{i + 1}" for i in range(self.num_hidden)]
            return z_names + x_names


    def get_intervention_list(self):
        x = self.get_features_train().cpu().numpy()

        """
            We'll intervene on the nodes corresponding to the self.intervention_index_list in x, 
            with the values that the samples have in the perc_idx percentile.
        """

        perc_idx = [25, 50, 75]

        """
            percentiles: values with which we do "do" operator 

            percentiles = np.percentile(x, [25, 50, 75], axis=0) -> output is (3, #nodes), so the 25, 50, and 75 percentile values along all dims of x
            percentiles = [np.mean(x)] (= np.mean(x, axis=0, keepdims=True))  -> output is (1, #nodes), so the mean along each dim
            percentiles = np.percentile(x, [10, 90], axis=0) -> output is (2, #nodes), the 10th and 90th percentile along each dim/node
        """

        percentiles = np.percentile(x, perc_idx, axis=0)
        int_list = []
        for i in self.intervention_index_list:
            """
                Select only those nodes which are in self.intervention_index_list, we only intervene on them
            """
            percentiles_i = percentiles[:, i-self.num_hidden]
            values_i = []
            for perc_name, perc_value in zip(perc_idx, percentiles_i):
                """
                    Get the corresponding percentiles for that node
                """
                values_i.append({"name": f"{perc_name}p", "value": perc_value})

            for value in values_i:
                """
                    Replace all values for this node('s samples) with the percentile values
                """
                value["value"] = round(value["value"], 2)
                value["index"] = i
                value["hidden_index"] = i - self.num_hidden # index to intervene when the node does not see the hidden variables
                int_list.append(value)

        return int_list

    def get_features_train(self):
        loader = self.get_dataloader_train(batch_size=self.num_samples())
        batch = next(iter(loader))

        return batch[0]

    def get_features_all(self):
        loader = self.get_dataloader_train(batch_size=self.num_samples())
        batch = next(iter(loader))
        # pdb.set_trace() # first appearance of data !!!
        if self.type == "torch":
            return self.base_distribution, self.correlations # torch.cat(batch, axis=0)
        elif self.type == "pyg":
            return self.base_distribution # batch.x.reshape(batch.num_graphs, -1)
        else:
            raise NotImplementedError(f"Type {self.type} not implemented")


    def _data_loader(self, dataset, batch_size, shuffle, num_workers=0):

        return tdata.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )

    def diameter(self):
        adjacency = self.adjacency(True).cpu().numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.Graph)
        try:
            diameter = nx.diameter(G)
        except:
            diameter = -1
        return diameter

    def longest_path_length(self):
        adjacency = self.adjacency(False).cpu().numpy()
        G = nx.from_numpy_matrix(adjacency, create_using=nx.DiGraph)
        try:
            longest_path_length = nx.algorithms.dag.dag_longest_path_length(G)
        except:
            longest_path_length = -1
        return int(longest_path_length)

    def get_ate_list(self):
        x = self.get_features_train().cpu().numpy()

        perc_idx = [25, 50, 75]

        percentiles = np.percentile(x, perc_idx, axis=0)
        int_list = []
        for i in self.intervention_index_list:
            percentiles_i = percentiles[:, i-self.num_hidden]
            values_i = []
            # values_i.append(
            #     {"name": "25_50", "a": percentiles_i[0], "b": percentiles_i[1]}
            # )
            # values_i.append(
            #     {"name": "25_75", "a": percentiles_i[0], "b": percentiles_i[2]}
            # )
            # values_i.append(
            #     {"name": "50_75", "a": percentiles_i[1], "b": percentiles_i[2]}
            # )

            for i_a,perc_a in enumerate(perc_idx):
                for i_b, perc_b in enumerate(perc_idx):
                    if perc_a < perc_b:
                        values_i.append(
                            {"name": f"{perc_a}_{perc_b}", "a": percentiles_i[i_a], "b": percentiles_i[i_b]}
                        )

            for value in values_i:
                value["a"] = round(value["a"], 2)
                value["b"] = round(value["b"], 2)
                value["index"] = i
                value["hidden_index"] = i - self.num_hidden
                int_list.append(value)

        return int_list

    def intervene(self, index, value, shape):
        """ So if we do intervene before sample, we get samples from the intervened distribution.
        But if we do sample before intervene, we get samples from the original distribution.        
        """
        index = index
        self.scm.intervene(index, value)
        x_int = self.scm.sample(shape)
        self.scm.stop_intervening(index)
        return x_int

    def compute_ate(self, index, a, b, num_samples=10000):
        index = index
        ate = self.scm.compute_ate(index, a, b, num_samples)
        return ate

    def compute_counterfactual(self, x_factual, index, value):
        index = index
        u = self.scm.transform.inv(x_factual)
        self.scm.intervene(index, value)
        x_cf = self.scm.transform(u)
        self.scm.stop_intervening(index)
        return x_cf

    def confounded_log_prob(self, x: torch.Tensor, x_full: torch.Tensor):
        return self.scm.confounded_log_prob(x, x_full, self.hidden_vars)    

    def log_prob(self, x):
        return self.scm.log_prob(x)
    
    def _loss(self, loss):
        if loss in ["default", "forward"]:
            return "forward"
        else:
            raise NotImplementedError(f"Wrong loss {loss}")

    def _split_dataset(self, dataset_raw):
        datasets = []

        for i, split_s in enumerate(self.split):
            num_samples = int(self._num_samples * split_s)
            if self.k_fold >= 0:
                seed = self.k_fold + i * 100
            else:
                seed = None

            dataset = SCMDataset(
                root_dir=self.root,
                num_samples=num_samples,
                hidden_vars=self.hidden_vars,
                scm=self.scm,
                name=self.name,
                sem_name=self.sem_name,
                use_edge_attr=self.use_edge_attr,
                seed=seed,
            )

            dataset.prepare_data()
            if i == 0:
                self.dataset = dataset
            datasets.append(dataset)

        return datasets

    def _get_dataset(self, num_samples, split_name):
        raise NotImplementedError

    def get_scaler(self, fit=True):

        scaler = self._get_scaler()
        self.scaler_transform = None
        if fit:
            x = self.get_features_train()
            scaler.fit(x, dims=self.dims_scaler)
            if self.scale in ["default", "std"]:
                self.scaler_transform = StandardTransform(
                    shift=x.mean(0), scale=x.std(0)
                )
                print("scaler_transform", self.scaler_transform)

        self.scaler = scaler

        return self.scaler

    def get_scaler_info(self):
        if self.scale in ["default", "std"]:
            return [("std", None)]
        else:
            raise NotImplementedError

    @property
    def dims_scaler(self):
        return (0,)

    def _get_dataset_raw(self):
        return None

    def _transform_dataset_pre_split(self, dataset_raw):
        return dataset_raw


    def post_process(self, x):
        offset = 0 if x.shape[1] == self.num_nodes_complete else self.num_hidden
        dims = list(map(lambda x: x - offset, self.dataset.binary_dims))

        mask = [x >= 0 for x in dims]
        dims = list(filter(lambda x: x >= 0, dims))

        if len(dims) > 0:
            x = x.clone()
            x[..., dims] = x[..., dims].floor().float()
            min_values = self.dataset.binary_min_values[mask]
            max_values = self.dataset.binary_max_values[mask]

            x[..., dims] = x[..., dims].floor().float()
            x[..., dims] = torch.clamp(x[..., dims], min=min_values, max=max_values)

        return x

    def check_tree(self):
        '''
        recall our definition of tree: a directed graph in which, from one of them, we can reach all others
        Returns: Bool, if the graph is a tree following our definition
        '''
        adjacency = self.get_severed_adjacency(False).numpy()

        # first, check if there are two nodes with no ingoing edges
        # if there are more than one, it is not a tree
        in_degrees = np.sum(adjacency, axis=1)
        if np.sum(in_degrees == 0) > 1:
            return False

        # now, check if we can reach all nodes from the one with no ingoing edges
        # to do so, get the ancestor matrix, sum(A^i for i in 1 to n-1)

        ancestors = ancestor_matrix(torch.tensor(adjacency)).numpy()
        # get the number of zeros in the lower triangular part of the ancestor matrix
        # if there are zeros, we cannot reach all nodes
        n = ancestors.shape[0]
        zero_count = 0
        for i in range(n):
            for j in range(i):  # j runs from 0 to i (exclusive)
                if ancestors[i, j] == 0:
                    zero_count += 1
        if zero_count > 0:
            return False
        return True

    def get_root_nodes(self, severed=False):
        '''
        Returns: List of root nodes
        '''

        adjacency = self.get_severed_adjacency(False).numpy() if severed else self.get_complete_adjacency(False).numpy()
        in_degrees = np.sum(adjacency, axis=1)
        return np.where(in_degrees == 0)[0]

