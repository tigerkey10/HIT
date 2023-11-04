
from typing import List
from torch import Tensor

from .hit import HIT

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ModelWrapper(nn.Module):
    def __init__(
        self,
        params,
        device='cpu'
    ) -> None:
        """ Model Wrapper """
        super().__init__()
        self.params = params

        self.device = device
        self.name = params['name']

        self.backbone = self.get_backbone()

        self.out_dim_in = self.params['dim_in']*2
        self.dropout = params['dropout']
        self.to_out = nn.Sequential(
                nn.Linear(self.out_dim_in, self.out_dim_in * 4),
                nn.GELU(),
                nn.Linear(self.out_dim_in * 4, self.params['y_dim']),
            )

    def forward(
        self,
        X,
        adj,
        idxs_g,
        idxs_d
        
    ) -> Tensor :
        emb, scores_attention = self.backbone(X, adj)
        out = []
        for g_i, d_i in zip(idxs_g, idxs_d):
            g_i, d_i = g_i.numpy(), d_i.numpy()
            out.append(torch.cat((emb[int(g_i)], emb[int(d_i)]), axis=-1))
        out = torch.stack(out, dim=0)
        out = self.to_out(out)

        return out


    def get_backbone(self):
        backbone = HIT(
                all_genes=self.params['data_inits']['all_genes'], 
                all_diseases=self.params['data_inits']['all_diseases'],
                input_dim=self.params['dim_in'], 
                embedding_dim=self.params['dim_hidden'], 
                num_heads=self.params['n_heads'], 
                integrated_gene_go_hyperedge_list=self.params['data_inits']['integrated_gene_go_hyperedge_list'],
                integrated_gene_go_node_list=self.params['data_inits']['integrated_gene_go_node_list'],
                hpo_do_gene_all_keys=self.params['data_inits']['hpo_do_gene_all_keys'],
                diseaseid_list_in_hpodogene=self.params['data_inits']['diseaseid_list_in_hpodogene'],
                device=self.device
            ).to(self.device)
        return backbone