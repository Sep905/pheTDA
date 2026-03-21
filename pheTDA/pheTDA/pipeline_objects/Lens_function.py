from .nn import AutoEncoder
from sklearn.manifold import MDS, TSNE , Isomap
from sklearn.decomposition import PCA
import umap


class Lens_function():
    def __init__(self, lens_name, lens_dict_params, dataset, distance_matrix):
        self.init_lens_and_project(lens_name, lens_dict_params, dataset, distance_matrix)

    def init_lens_and_project(self, lens_name, lens_dict_params, dataset, distance_matrix):
        if lens_name == "PCA":
            self.lens_function = PCA(n_components=lens_dict_params['projection_dimension'], 
                                     random_state=lens_dict_params['seed'])
            self.projections = self.lens_function.fit_transform(dataset)

        elif lens_name in ['MDS','Isomap','t-SNE','UMAP']:

            if lens_name == "MDS":
                self.lens_function = MDS(n_components=lens_dict_params['projection_dimension'], 
                                         random_state=lens_dict_params['seed'], 
                                         metric_mds=lens_dict_params['metric'], 
                                         metric ="precomputed",
                                         init='random',
                                         n_init = lens_dict_params['n_init'],
                                         max_iter=lens_dict_params['max_it_mds'], 
                                         eps=lens_dict_params['eps_mds'],
                                         n_jobs=1)

            elif lens_name == "Isomap":
                self.lens_function = Isomap(n_components=lens_dict_params['projection_dimension'], 
                                            path_method="D",
                                            metric="precomputed",
                                            n_neighbors = lens_dict_params['n_neighbors_isomap'],
                                            n_jobs=1)
                    
            elif lens_name == "t-SNE":
                self.lens_function = TSNE(n_components=lens_dict_params['projection_dimension'], 
                                          random_state=lens_dict_params['seed'], 
                                          init="random", 
                                          metric="precomputed",
                                          perplexity = lens_dict_params['perplexity'],  
                                          learning_rate = lens_dict_params['learning_rate_tsne'], 
                                          max_iter =  lens_dict_params['n_iter'],
                                          n_jobs=1)

            elif lens_name == "UMAP":
                self.lens_function = umap.UMAP(n_components=lens_dict_params['projection_dimension'],
                                               random_state=lens_dict_params['seed'], 
                                               metric="precomputed",
                                               n_neighbors = lens_dict_params['n_neighbors_umap'], 
                                               min_dist = lens_dict_params['min_dist'],
                                               n_jobs=1)
            
            self.projections = self.lens_function.fit_transform(distance_matrix)

        elif lens_name == "AutoEncoder":
            self.lens_function = AutoEncoder(input_dim = dataset.shape[1],
                                             num_layers = lens_dict_params['num_layers'],   
                                             use_batchnorm = lens_dict_params['use_batchnorm'],  
                                             use_dropout = lens_dict_params['use_dropout'],
                                             dropout_prob = lens_dict_params['dropout_prob'],   
                                             activation_function = lens_dict_params['activation_function'], 
                                             learning_rate = lens_dict_params['learning_rate_ae'],  
                                             w_decay = lens_dict_params['w_decay'],   
                                             batch_size = lens_dict_params['batch_size'],
                                             epochs = lens_dict_params['epochs'],
                                             random_state = lens_dict_params['seed'])

            self.projections = self.lens_function.fit_transform(dataset).detach().cpu().numpy()
