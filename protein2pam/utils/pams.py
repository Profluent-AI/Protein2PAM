"""
This module provides functionality for PAM (Protospacer Adjacent Motif) objects.
"""

import sys
import json
import pandas as pd
import logomaker
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class PAM:
    """Represents a PAM object."""
    def __init__(self, results, seq_idx):
        """Initialize a PAM object."""
        self.model_name = results["model_name"]
        self.sequence = results["sequences"][seq_idx]
        self.pam_prediction = results["predictions"][seq_idx]
        self.percent_identity = 100*max(results["percent_identities"][seq_idx]) if "percent_identities" in results else None
        self.confidence_score = results["confidence_scores"][seq_idx] if "confidence_scores" in results else None
        self.confidence_level = self.set_conf_level()        
        self.pam_side = "downstream" if "cas9" in self.model_name else "upstream"
        self.is_downstream = self.pam_side=="downstream"
        self.consensus_pam = self.consensus_pam()
        self.pam_nucleotides = list("ACGT")
        self.pam_positions = list(range(1,11)) 
        if not self.is_downstream:
            self.pam_positions = [-1 * _ for _ in list(range(1,11))][::-1]
    

    def __repr__(self):
        """Returns a string representation of the PAM object."""
        return (            
            f"model_name='{self.model_name}',\n"
            f"pam_side='{self.pam_side}',\n"
            f"pam_nucleotides='{self.pam_nucleotides}',\n"
            f"pam_positions='{self.pam_positions}',\n"
            f"sequence='{self.sequence}',\n"
            f"pam_prediction='{self.pam_prediction}',\n"
            f"percent_identity={self.percent_identity},\n"
            f"confidence_score={self.confidence_score},\n"
            f"confidence_level='{self.confidence_level}',\n"
            f"consensus_pam='{self.consensus_pam}'\n"
        )

    def set_conf_level(self) -> bool:
        """Checks if the PAM has a high confidence level based on the confidence score and level."""
        if self.confidence_score is None: 
            return None
        elif self.confidence_score > 0.80:
            return "high"
        elif self.confidence_score > 0.70:
            return "med"
        else:
            return "low"

    def update_pam_info(self, key: str, value):
        """Updates or adds a key-value pair to the PAM information dictionary."""
        self.pam_info[key] = value

    def get_pam_info(self, key: str):
        """Retrieves a value from the PAM information dictionary by key."""
        return self.pam_info.get(key, None)
    
    def consensus_pam(self, min_info: float = 0.70) -> str:
        """
        Returns a consensus PAM associated with an information matrix.
        We say a position is N if the total information is <1 bit and the maximum probability of any base is <70%.
        """
        nts = [
            "N" if i.max() < min_info else "ACGT"[i.argmax()] for i in self.pam_prediction
        ]
        return "".join(nts).rstrip("N") if self.pam_side == "downstream" else "".join(nts).lstrip("N")    
    

    def plot_logo(self, file=None, show=True, side="downstream", title=None):
        """Plots a logo of the PAM prediction."""
        plt.rcParams['figure.dpi'] = 150   

        pam_df = pd.DataFrame(self.pam_prediction)
        pam_df.columns = ["A", "C", "G", "T"]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))
        pam_logo = logomaker.Logo(pam_df, ax=ax)
        
        pam_logo.style_spines(visible=False)
        pam_logo.style_spines(spines=["left", "bottom"], visible=True)
        pam_logo.style_xticks(fmt="%d", anchor=0)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        
        if side == "downstream":
            pam_logo.ax.set_xticklabels([_ for _ in range(1, len(pam_df) + 1)], size=12)
        else:
            pam_logo.ax.set_xticklabels([-_ for _ in range(1, len(pam_df) + 1)][::-1], size=12)
        
        for y in pam_logo.ax.get_yticklabels():
            y.set_size(12)
        
        pam_logo.ax.set_ylabel("bits", size=15)
        ax.set_ylim(0, 2)
        
        if title is not None:
            ax.set_title(title)
        
        fig.tight_layout() 
        if file:
            fig.savefig(file)

        if show:
            plt.show()
            
        plt.close(fig)
        
     