import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore
import pandas as pd
import warnings

# warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict

    def get_features(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1):

        assert len(timesteps) == 1

        dfs = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            gmm_dist, latent_probs = model.get_features(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples)

            timestep_range_x = np.array([timesteps[0] - self.max_ht, timesteps[0]])
            timestep_future = np.array([timesteps[0] + 1, timesteps[0] + ph])


            latent_np = latent_probs.detach().cpu().numpy()
            latent_probs = {}
            gmm_means = gmm_dist.mus
            gmm_covs = gmm_dist.get_covariance_matrix()
            gmm_means_np = gmm_means.detach().cpu().numpy()
            gmm_covs_np = gmm_covs.detach().cpu().numpy()

            state_labels = []
            for k, v in self.state[str(node_type)].items():
                tps = [k + "_" + v[i] for i in range(len(v))]
                state_labels += tps

            control_mask = []
            control_labels = []
            for st_labels in state_labels:
                if ("acceleration" in st_labels or "heading_d" in st_labels):
                    control_mask.append(True)
                    control_labels.append(st_labels + "_true")
                else:
                    control_mask.append(False)
            
            multi_index = []

            nodes_strs = [str(nodes[i]) for i in range(len(nodes))]
            
            for j in range(1, 3):
                multi_index += [(f'gmm_mean_{j}', i) for i in range(ph)]
                multi_index += [(f'gmm_cov_{j}', i) for i in range(ph)]
            multi_index += [("gmm_corr", i) for i in range(ph)]
            multi_index += [("latent_prob", -1)]

            col_index = []
            for node in nodes_strs:
                col_index += [(node, k) for k in range(latent_np.shape[-1])]
            
            col_idx = pd.MultiIndex.from_tuples(col_index)

            multi_index_state = []
            for st_label in state_labels:
                tps = [(st_label, i) for i in range(self.max_ht + 1)]
                multi_index_state += tps
            for c_label in control_labels:
                tps = [(c_label, i) for i in range(ph)]
                multi_index_state += tps
            
            m_idx = pd.MultiIndex.from_tuples(multi_index)
            m_idx_state = pd.MultiIndex.from_tuples(multi_index_state)

            df_pred = pd.DataFrame(index=col_idx, columns=m_idx)
            df_pred = df_pred.astype(object)

            df_state = pd.DataFrame(index=nodes_strs, columns=m_idx_state)

            for i in range(len(nodes)):
                node_states = nodes[i].get(timestep_range_x, self.state[node_type]) # (max_ht+1, |state|)
                node_states_ft = nodes[i].get(timestep_future, self.state[node_type]) # (ph, |state|)
                node_control_ft = node_states_ft[:, control_mask] # (ph, |U|)
                node_latent = latent_np[i,:,:].squeeze() # (|Z|,)
                node_means = gmm_means_np[:,i,:,:,:].squeeze() # (ph, |Z|, |U|)
                means_list = [node_means[i,:,:] for i in range(ph)]
                node_covs = gmm_covs_np[:,i,:,:,:,:].squeeze() # (ph, |Z|, |U|, |U|)
                covs_list = [node_covs[i,:,:,:] for i in range(ph)]
                for l in range(latent_np.shape[-1]):
                    for j in range(ph):
                        for k in range(1, 3):
                            df_pred.loc[str(nodes[i]), l][f'gmm_mean_{k}', j] = means_list[j][l,k-1]
                            df_pred.loc[str(nodes[i]), l][f'gmm_cov_{k}', j] = covs_list[j][l,k-1, k-1]
                        df_pred.loc[str(nodes[i]), l][f'gmm_corr', j] = covs_list[j][l,0, 1]
                    df_pred.loc[str(nodes[i]), l]["latent_prob", -1] = node_latent[l]
                for j in range(len(state_labels)):
                    for k in range(self.max_ht + 1):
                        df_state.loc[str(nodes[i])][state_labels[j], k] = node_states[k,j]
                for j in range(len(control_labels)):
                    for k in range(ph):
                        df_state.loc[str(nodes[i])][control_labels[j], k] = node_control_ft[k,j]

            dfs[node_type] = [df_state.copy(), df_pred.copy()]

        return dfs
