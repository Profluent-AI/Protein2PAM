"""
This module contains the Oracle class, which is used to evaluate the performance of a model.
"""

import os
import logging
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from Bio import SeqIO
from Levenshtein import distance as ld

from protein2pam.models.getter import get_model
from protein2pam.models.esm2 import ESMPlusHead
from protein2pam.utils.common import prob_to_info
from protein2pam.utils.pams import PAM

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.amp")


class Oracle(ABC):
    """Abstract base class for all Oracle classes."""
    @abstractmethod
    def evaluate(self, proteins, ten_nn_pct_id, train_seqs):
        """Evaluates a batch of proteins and returns a dict mapping score name to a tensor of
        scores for the protein batch.

        Can return more than one score.
        """
        raise NotImplementedError


class PAMOracle(Oracle):
    """Oracle for PAM prediction."""
    def __init__(
        self,
        model_name: str = None,
        data_dir: str = None,
        uncertainty_quantification: bool = True      
    ):

        # setup inputs
        self.model_name = model_name
        self.data_dir = data_dir
        self.family_name = self.model_name.split("_")[0]
        self.validate_model()
        self.fetch_db_files()

        # init pam model
        self.f = MarginalPAMOracle(
            model_path = self.model_path,
            config_path = self.model_config_path,
            model_name = self.model_name
        )
        
        # init confidence model
        self.uq = None
        if uncertainty_quantification:         
            self.uq = MarginalPAMInfoSimOracle(
                model_path = self.uq_model_path,                
                config_path = self.uq_config_path,
                fasta_path = self.fasta_path
            )

    def get_db_path(self, file_name):
        """Get the path to a database file."""
        import importlib.resources
        with importlib.resources.path("protein2pam.data", file_name) as file_path:
            return str(file_path)

    def fetch_db_files(self):  
        """Fetch the database files."""
        self.model_path = os.path.join(self.data_dir, self.model_name+".pt")   
        self.fasta_path = os.path.join(self.data_dir, self.model_name+".fasta")
        self.model_config_path = os.path.join(self.data_dir, self.model_name+".yml")
        self.uq_model_path = os.path.join(self.data_dir, self.family_name+"_uq.pt")
        self.uq_config_path = os.path.join(self.data_dir, self.family_name+"_uq.yml")
        for file_path in [self.model_path, self.fasta_path, self.model_config_path, self.uq_model_path, self.uq_config_path]:
            if not os.path.exists(file_path):
                error_message = (
                    f"One or more required database files are missing in the directory: '{self.data_dir}'.\n"
                    f"Missing files:\n{file_path}\n"
                    f"Please ensure that the required files are present before proceeding."
                )                
                logger.error(error_message)
                return

    def validate_model(self):
        """Validate the model."""
        available_models = ("cas8", "cas9", "cas12", "cas9_full", "cas9_full_nolit", "cas9_pid_nolit", "cas9_pid_nme", "cas12_nolit")
        if self.model_name not in available_models:
            error_message = (
                f"Invalid model specified: '{self.model_name}'.\n"
                "Please choose a valid model from the following options:\n"
                + "\n".join(f"- {model}" for model in available_models)
            )
            logging.error(error_message)
            raise ValueError(error_message)

    def cpu(self):
        """Move the model to the CPU."""
        self.f.cpu()
        if self.uq is not None:
            self.uq.cpu()
        return self

    def cuda(self):
        """Move the model to the GPU."""
        self.f.cuda()
        if self.uq is not None:
            self.uq.cuda()
        return self

    def evaluate(self, proteins, ten_nn_pct_id=None, train_seqs=None):
        """Evaluate the model."""
        if isinstance(proteins, str):
            proteins = [proteins]
        result = self.f.evaluate(proteins)
        if self.uq is not None:
            uq_result = self.uq.evaluate(proteins, ten_nn_pct_id, train_seqs)
            result.update(uq_result)
        
        pams = [PAM(result, i) for i in range(len(result["sequences"]))] 
        return pams
    
            
class MarginalPAMOracle(Oracle):
    """Oracle for marginal PAM prediction."""
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        config_path: str = None,
        suffix: str = "",
    ):
        super().__init__()
        self.model_name = model_name
        self.model: ESMPlusHead = get_model(config_path, path = model_path)

    def cpu(self):
        """Move the model to the CPU."""
        self.model.cpu()
        return self

    def cuda(self):
        """Move the model to the GPU."""
        self.model.cuda()
        return self

    def evaluate(self, proteins, target_prob = None):
        """Evaluate the model."""
        sequences = [proteins] if isinstance(proteins, str) else proteins
        result = {}
        with torch.no_grad():
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'            
            dtype = torch.float32
            with torch.amp.autocast(device_type=device_type, dtype=dtype):                
                outputs = self.model(sequences)
                logp = F.log_softmax(outputs, dim=-1)
                info = prob_to_info(torch.exp(logp), logp)
                result["predictions"] = info.detach().cpu().numpy()            
                result["model_name"] = self.model_name
                result["sequences"] = proteins                       
        return result

            
class MarginalPAMInfoSimOracle(Oracle):
    """Oracle for marginal PAM prediction with information similarity."""
    def __init__(
        self,        
        model_path: str = None,                              
        config_path: str = None,            
        fasta_path: str = None
    ):
        super().__init__()
        self.model: ESMPlusHead = get_model(config_path, path=model_path)
        self.train_seqs = [
            str(s.seq) for s in SeqIO.parse(fasta_path, format="fasta")
        ]

    def cpu(self):
        """Move the model to the CPU."""
        self.model.cpu()
        return self

    def cuda(self):
        """Move the model to the GPU."""
        self.model.cuda()
        return self

    def evaluate(self, proteins, ten_nn_pct_id, train_seqs):
        """Evaluate the model."""
        sequences = [proteins] if isinstance(proteins, str) else proteins

        if train_seqs is None:
            train_seqs = self.train_seqs

        if train_seqs is not None and ten_nn_pct_id is None:
            ten_nn_pct_id = []
            for seq in sequences:
                pct_id = [1 - ld(seq, s) / max(len(seq), len(s)) for s in train_seqs]
                ten_nn_pct_id.append(sorted(pct_id, reverse=True)[:10])
            ten_nn_pct_id = torch.tensor(ten_nn_pct_id)

        result = {}
        with torch.no_grad():
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float32
            with torch.amp.autocast(device_type=device_type, dtype=dtype):
                if ten_nn_pct_id is not None:
                    result["percent_identities"] = ten_nn_pct_id
                    if not isinstance(ten_nn_pct_id, torch.Tensor):
                        ten_nn_pct_id = torch.tensor(ten_nn_pct_id)
                    expected = (len(sequences), 10)
                    assert (
                        ten_nn_pct_id.shape == expected
                    ), f"Expected ten_nn_pct_id to have shape {expected}, but got {ten_nn_pct_id.shape}"
                result["confidence_scores"] = self.model(
                    sequences, ten_nn_pct_id
                ).flatten()

        result = {k: v.detach().cpu().numpy() for k, v in result.items()}
        return result



