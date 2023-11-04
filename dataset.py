from collections import OrderedDict
from sklearn import preprocessing 
from utils import seed_everything

import obonet
import pandas as pd
import json
import pickle
import torch
import numpy as np 
import hypernetx as hnx
import numpy as np 
import math
import time

class Dataset:
    def __init__(
        self,
        go=True,
        hpo=True,
        do=True,
        gg=True,
        dg=True,
        seed=0
    ):
        seed_everything(seed)
        self.go = go
        self.do = do
        self.hpo = hpo
        self.gg = gg
        self.dg = dg

        print('- load datasets', end = '\t')
        start = time.time()
        self.load_dataset()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('- load the information of unique set', end = '\t')
        start = time.time()
        self.load_unique_info()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('-- generate the gene index rule', end = '\t')
        start = time.time()
        self.GF_indexrule()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('-- generate the disease index rule', end = '\t')
        start = time.time()
        self.DF_indexrule()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('--- Construct the hypergraph for gene', end = '\t')
        start = time.time()
        self.construct_hypergraph_g()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('--- Construct the hypergraph for disease', end = '\t')
        start = time.time()
        self.construct_hypergraph_d()
        end = time.time()
        print(f"{end - start:.5f} sec")

        print('---- Set the labels', end = '\t')
        start = time.time()
        self.set_tbga()
        end = time.time()
        print(f"{end - start:.5f} sec")

    def load_dataset(self):
        # Dataset load
        with open("./datasets/datasets.pkl", "rb") as f:
            loaded_tensor = pickle.load(f)

        self.df_tbga = loaded_tensor['tbga']
        self.gene_gene_adj = loaded_tensor['gene_gene']
        self.gene_go_adj = loaded_tensor['gene_go']
        self.dis_hpo_adj = loaded_tensor['disease_hpo']
        self.dis_do_adj = loaded_tensor['disease_do']

    def load_unique_info(self):
        # Unique node set load
        with open("./datasets/unique_node_sets.pkl", "rb") as f:
            loaded_tensor = pickle.load(f)
    
        self.unique_genes_set = loaded_tensor['unique_genes_set']
        self.unique_go_set = loaded_tensor['unique_go_set']
        self.unique_disease_set = loaded_tensor['unique_disease_set']
        self.unique_hpo_set = loaded_tensor['unique_hpo_set']
        self.unique_do_set = loaded_tensor['unique_do_set']


    def GF_indexrule(self):
        self.unique_genes_list = list(self.unique_genes_set)
        self.unique_genes_list.sort()
        geneid_to_index_rule = OrderedDict((i,idx) for idx, i in enumerate(self.unique_genes_list))
        self.len_g = len(geneid_to_index_rule)

        if self.go:
            self.unique_go_list = list(self.unique_go_set)            
            self.unique_go_list.sort()
            goid_to_index_rule = OrderedDict((i,idx+self.len_g) for idx, i in enumerate(self.unique_go_list))
            self.len_g += len(goid_to_index_rule)
        else:
            goid_to_index_rule = None

        self.geneid_to_index_rule = geneid_to_index_rule
        self.goid_to_index_rule = goid_to_index_rule

    def DF_indexrule(self):

        self.total_diseaseid_list = list(self.unique_disease_set)
        self.total_diseaseid_list.sort()
        diseaseid_to_index_rule = OrderedDict((i,idx+self.len_g) for idx, i in enumerate(self.total_diseaseid_list))
        self.len_d = self.len_g + len(diseaseid_to_index_rule)

        if self.hpo:
            self.unique_hpo_list = list(self.unique_hpo_set)
            self.unique_hpo_list.sort()
            hpoid_to_index_rule = OrderedDict((i,idx+self.len_d) for idx, i in enumerate(self.unique_hpo_list))
            self.len_d += len(hpoid_to_index_rule)
        else:
            hpoid_to_index_rule = None

        if self.do:
            self.unique_do_list = list(self.unique_do_set)
            self.unique_do_list.sort()
            doid_to_index_rule = OrderedDict((i,idx+self.len_d) for idx, i in enumerate(self.unique_do_list))
            self.len_d += len(doid_to_index_rule)
        else:
            doid_to_index_rule = None

        self.diseaseid_to_index_rule = diseaseid_to_index_rule
        self.hpoid_to_index_rule = hpoid_to_index_rule
        self.doid_to_index_rule = doid_to_index_rule

    def construct_hypergraph_g(self):
        self.all_genes = set(self.geneid_to_index_rule.values())

        # go_list = list(self.goid_to_index_rule.values())
        self.gene_gene_adj.iloc[:,0] = [self.geneid_to_index_rule[i] for i in self.gene_gene_adj.iloc[:,0].values]
        self.gene_gene_adj.iloc[:,1] = [self.geneid_to_index_rule[i] for i in self.gene_gene_adj.iloc[:,1].values]
        self.generate_hg_gg()
        
        gene_gene = np.array(self.gene_gene_adj).T
        # self.geneid_set = set(gene_gene[0]).union(set(gene_gene[1]))
        self.geneid_set = set(gene_gene[1])

        if self.go:
            self.gene_go_adj.iloc[:,0] = [self.geneid_to_index_rule[i] for i in self.gene_go_adj.iloc[:,0].values]
            self.gene_go_adj.iloc[:,1] = [self.goid_to_index_rule[i] for i in self.gene_go_adj.iloc[:,1].values]
            self.generate_hg_go()
            go = np.array(self.gene_go_adj).T
            gene_go_list = list(set(go[0]))
            self.geneid_set = set(gene_go_list).union(set(gene_gene[1]))
        
        self.geneid_list = list(self.geneid_set)
        self.df_tbga.iloc[:,1] = [self.geneid_to_index_rule[i] for i in self.df_tbga.iloc[:,1].values]

        self.integrated_genes_hyperedge_list = list(self.hyperedges_gene_gene.keys())
        gene_integrated = hnx.Hypergraph(self.hyperedges_gene_gene)
        self.gene_integrated_incidence = gene_integrated.incidence_matrix()
        self.gene_incidence_matrix = self.gene_integrated_incidence.toarray()

    def generate_hg_gg(self):
        gene_gene = np.array(self.gene_gene_adj).T
        neighbors = {node : set() for node in np.unique(gene_gene[0])}
        for node1, node2 in gene_gene.T : 
            neighbors[node1].add(node2)
        self.hyperedges_gene_gene = {node:neighbors[node] for node in neighbors}

    def generate_hg_go(self):
        # Gene-GO incidence matrix generation
        go = np.array(self.gene_go_adj).T
        neighbors = {node : set() for node in np.unique(go[1])}
        for node1, node2 in go.T : 
            neighbors[node2].add(node1)
        hyperedges_geneGO = {node : neighbors[node] for node in neighbors}

        self.hyperedges_gene_gene.update(hyperedges_geneGO)

    def construct_hypergraph_d(self):
        self.all_diseases = set(self.diseaseid_to_index_rule.values())
        
        self.df_tbga.iloc[:,2] = [self.diseaseid_to_index_rule[i] for i in self.df_tbga.iloc[:,2].values]
        self.generate_hg_tbga()
        self.diseaseid_set = set(self.df_tbga_gene_dis_array[1])

        if self.hpo:
            self.dis_hpo_adj.iloc[:,0] = [self.diseaseid_to_index_rule[i] for i in self.dis_hpo_adj.iloc[:,0].values]
            self.dis_hpo_adj.iloc[:,1] = [self.hpoid_to_index_rule[i] for i in self.dis_hpo_adj.iloc[:,1].values]
            self.generate_hg_hpo()
            self.diseaseid_set = self.diseaseid_set.union(set(self.dis_hpo_adj.iloc[:, 0].values))

        if self.do:
            self.dis_do_adj.iloc[:,0] = [self.diseaseid_to_index_rule[x] for x in self.dis_do_adj.iloc[:,0].values]
            self.dis_do_adj.iloc[:,1] = [self.doid_to_index_rule[x] for x in self.dis_do_adj.iloc[:,1].values]
            self.generate_hg_do()
            self.diseaseid_set = self.diseaseid_set.union(set(self.dis_do_adj.iloc[:, 0].values))

        disease_integrated = hnx.Hypergraph(self.hyperedges_tbga)
        self.disease_integrated_incidence = disease_integrated.incidence_matrix()      

        # integrated disease flow incidence matrix
        self.disease_integrated_incidence_matrix = self.disease_integrated_incidence.toarray()

        # disease-hpo, do, gene에서 사용한 전체 hpo, do, gene 인덱스 리스트
        self.hpo_do_gene_all_keys = list(set(self.hyperedges_tbga.keys()))
        # hpo do gene hypergraph 안에 사용된 disease id들
        self.diseaseid_list = list(self.diseaseid_set)

    def generate_hg_tbga(self):
        # TBGA Gene-Disease incidence matrix generation
        self.df_tbga_gene_dis_array = np.array(self.df_tbga[self.df_tbga.iloc[:,0] != 'NA'].iloc[:,[1,2]]).T
        neighbors = {node : set() for node in np.unique(self.df_tbga_gene_dis_array[0])}
        for node1, node2 in self.df_tbga_gene_dis_array.T : 
            neighbors[node1].add(node2)
        self.hyperedges_tbga = {node : neighbors[node] for node in neighbors}
    
    def generate_hg_hpo(self):
        # Disease-HPO incidence matrix generation
        disease_hpo = np.array(self.dis_hpo_adj).T
        neighbors = {node : set() for node in np.unique(disease_hpo[1])}
        for node1, node2 in disease_hpo.T : 
            neighbors[node2].add(node1)
        hyperedges_diseaseHPO = {node:neighbors[node] for node in neighbors}
        
        self.hyperedges_tbga.update(hyperedges_diseaseHPO)

    def generate_hg_do(self):
        # Disease-HPO incidence matrix generation
        disease_do = np.array(self.dis_do_adj).T
        neighbors = {node : set() for node in np.unique(disease_do[1])}
        for node1, node2 in disease_do.T : 
            neighbors[node2].add(node1)
        hyperedges_diseaseDO = {node:neighbors[node] for node in neighbors}

        self.hyperedges_tbga.update(hyperedges_diseaseDO)
        
    def set_tbga(self):
        self.df_tbga['relation'] = self.df_tbga['relation'].map({'NA':0, 'biomarker':1, 'genomic_alterations':1, 'therapeutic':2})
        self.df_tbga = self.df_tbga.rename(columns={'relation':'relation', 'geneid':'geneId', 'diseaseid': 'diseaseId'})
        self.label = self.df_tbga[['geneId', 'diseaseId', 'relation']]
        self.label = self.label.values

    def get_data_inits(self):
        self.integrated_genes_hyperedge_list.sort()
        self.geneid_list.sort()
        self.hpo_do_gene_all_keys.sort()
        self.diseaseid_list.sort()
        
        return {
            'all_genes' : self.all_genes,
            'all_diseases' : self.all_diseases,
            'integrated_gene_go_hyperedge_list' : self.integrated_genes_hyperedge_list,
            'integrated_gene_go_node_list' : self.geneid_list, 
            'hpo_do_gene_all_keys' : self.hpo_do_gene_all_keys,
            'diseaseid_list_in_hpodogene' : self.diseaseid_list,
        }
    
    def get_data_forwards(self):
        return {
            'gene_total_incidence_matrix' : torch.tensor(self.gene_incidence_matrix),
            'disease_integrated_incidence' : torch.tensor(self.disease_integrated_incidence_matrix)
        }