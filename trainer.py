from typing import Dict, Tuple, List
from torch.utils.data import DataLoader
from einops import rearrange
from utils import get_DataLoader, split_data
from utils import seed_everything, count_parameters, get_optimizer
from models.modelwrapper import ModelWrapper
from sklearn.metrics import accuracy_score, f1_score
from dataset import Dataset
import torch
import torch.nn as nn
import numpy as np
import copy
import pickle


# Constants
MODEL_NAME_MAP = {
    'HIT':Dataset,
}

class Trainer:
    def __init__(
        self,
        params,
        seed=315,
        device='cpu',
    ):
        super().__init__()        
        seed_everything(seed)
        self.device = device
        self.seed = seed
        self.params = params
        self.name = self.params['name']
        self.setup_datasets()
        self.define_model()
        self.best_model_state = self.model.state_dict()
        self.loss = nn.CrossEntropyLoss()
            
    def setup_datasets(self):
        dataset_class = MODEL_NAME_MAP.get(self.params['name'].upper())
        if dataset_class is None:
            raise ValueError(f"Invalid model name: {self.params['name']}")
            
        self.data = dataset_class(go=self.params['go'], hpo=self.params['hpo'], do=self.params['do'], seed=self.seed)
        data_forwards = self.data.get_data_forwards()
        self.X = data_forwards['gene_total_incidence_matrix'].to(self.device)
        self.adj = data_forwards['disease_integrated_incidence'].to(self.device)


    def define_model(self):
        self.params['data_inits'] = self.data.get_data_inits()
    
        self.model = ModelWrapper(self.params, self.device).to(self.device)

        idxs_train, idxs_valid, idxs_test = split_data(self.data.label, seed=self.seed)
        self.loader_train = get_DataLoader(self.data.label[idxs_train, :], self.params['batch_size'], True)
        self.loader_valid = get_DataLoader(self.data.label[idxs_valid, :], self.params['batch_size'], False)
        self.loader_test = get_DataLoader(self.data.label[idxs_test, :], self.params['batch_size'], False)

    def get_count_params(self, model=None, name=None):
        return count_parameters(model) if model is not None else count_parameters(self.model)

    def set_optimizer(self,optimizer='adamw',lr=1e-4,w_decay=None):
        self.optimizer = get_optimizer(self.model.parameters(), optimizer, lr, w_decay)

    def fit(
            self,
            epochs=100,
            optimizer='adamw',
            lr=1e-4,
            w_decay=None, 
            verbose=True,
            earlystop=None,
    ):
        """ fit """
        self.set_optimizer(optimizer, lr, w_decay)
        earlystop = epochs if earlystop is None else earlystop
        s_best, patience = 0, 1
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        for e in range(1, epochs+1):
            if verbose:
                print('\r---------------------------------------------------------------------------------------------------------------')
                print('| Data  |  Epoch  | Loss      accuracy   f1-micro   f1-macro    f1-NA    f1-Biomarker    f1-Therapeutic   | BEST')

            self.train = True
            self.model.train()
            l_train, s_train = self.step_batch(self.loader_train)
            if verbose:
                print( f'\r| Train | {e}/{epochs} | {l_train:.6f}    {s_train[0]:.6f}   {s_train[1]:.6f}    {s_train[2]:.6f}   {s_train[4]:.6f}   {s_train[5]:.6f}   {s_train[6]:.6f}   |')


            with torch.no_grad():
                self.train = False
                self.model.eval()
                l_valid, s_valid = self.step_batch(self.loader_valid)
                isBest = True if s_valid[2] > s_best else False # best as Macro F1 Score
                if verbose:
                    best_str = '->Best!' if isBest else f'[{patience}/{earlystop}]'
                    print( f'\r| Valid | {e}/{epochs} | {l_valid:.6f}    {s_valid[0]:.6f}   {s_valid[1]:.6f}    {s_valid[2]:.6f}   {s_valid[4]:.6f}   {s_valid[5]:.6f}   {s_valid[6]:.6f}   |', best_str)
                 
                l_test, s_test = self.step_batch(self.loader_test)
                if verbose:
                    print( f'\r| Test  | {e}/{epochs} | {l_test:.6f}    {s_test[0]:.6f}   {s_test[1]:.6f}    {s_test[2]:.6f}   {s_test[4]:.6f}   {s_test[5]:.6f}   {s_test[6]:.6f}    |')

            isBest = True if s_valid[2] > s_best else False  # best as Macro F1 Score
            if isBest:
                e_best, s_best, s_test_best_micro, s_test_best_macro, s_test_best_NA, s_test_best_biomarker, s_test_best_thera, patience = e, s_valid[2], s_test[1], s_test[2], s_test[4], s_test[5], s_test[6], 1
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                #torch.save(self.model.state_dict(), self.saved_model_path)

            else:
                patience += 1

            if verbose:
                print('-'*100)
                print( f'| BEST  |Epoch:{e_best}| Valid Score: {s_best:.6f}   Test Score(micro): {s_test_best_micro:.6f}   Test Score(macro): {s_test_best_macro:.6f}')
                print( f'| BEST  |Epoch:{e_best}| Test Score(NA): {s_test_best_NA:.6f}   Test Score(Biomarker): {s_test_best_biomarker:.6f}   Test Score(Therapeutic): {s_test_best_thera:.6f}  !')
                print('-'*100)
                
            if patience >= earlystop:
                break
        self.model.load_state_dict(self.best_model_state)
        l_test, s_test = self.step_batch(self.loader_test)
        if verbose:
            print('Valid Score %.4f (epoch : %d)' % (s_best, e_best))
            print('Test Score %.4f %.4f %.4f %.4f %.4f %.4f %4f' % (s_test[0], s_test[1], s_test[2], s_test[3], s_test[4], s_test[5], s_test[6]))
            print('params : ', count_parameters(self.model))

    def load_saved_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
    
    def step_batch(self, dataloader: DataLoader) -> Tuple[float, Tuple]:
        y_labels, y_preds = torch.empty(0).to(self.device), torch.empty(0).to(self.device)
        loss_total = 0.0
        
        for idx, (idxs_g, idxs_d, y_label) in enumerate(dataloader, 1):
            if self.train:
                self.optimizer.zero_grad()

            y_pred = self.model(self.X, self.adj, idxs_g, idxs_d)
            
            loss = self.loss(y_pred, y_label.to(self.device))
            
            if self.train:
                loss.backward()
                self.optimizer.step()

            y_preds = torch.cat([y_preds, y_pred], dim=0)
            y_labels = torch.cat([y_labels, y_label.to(self.device)], dim=0)
            loss_total += loss.item()
            print(f'\r [{idx}/{len(dataloader)}] loss : {loss.item():.4f}', end='')

        scores = self.get_score(y_preds, y_labels)
        loss_avg = loss_total / len(dataloader)
        return loss_avg, scores
    
    def get_score(
        self,
        y_pred,
        y_label,
    ):

        y_pred_labels = torch.argmax(y_pred, dim=1)
        acc = accuracy_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy())
        f1_micro = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average='micro')
        f1_macro = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average='macro')
        f1_weighted = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average='weighted')
        
        f1_NA = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average=None)[0]
        f1_biomarker = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average=None)[1]
        f1_therapeutic = f1_score(y_label.cpu().detach().numpy(), y_pred_labels.cpu().detach().numpy(), average=None)[2]
        return acc, f1_micro, f1_macro, f1_weighted, f1_NA, f1_biomarker, f1_therapeutic 