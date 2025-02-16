

class Config:
    def __init__(self):
      self.transfer = "yes"
      self.selected_code = "en_transfer_learning.py" #The code to execute
      
      if self.transfer == "yes":
        self.epochs_a = 200
        self.epochs_b = 200
        self.hidden_size = 20
        self.batch_size_a = 300
        self.batch_size_b = 150
        self.weight_decay_a = 0
        self.weight_decay_b = 0
        self.base_lr_a = 0.0002
        self.max_lr_a = 0.009
        self.base_lr_b = 0.004
        self.max_lr_b = 0.05
        self.num_features_to_keep = 23
        self.source_data = "datasets/izmir_iod_dataset.xlsx"
        self.target_data = "datasets/ankara_iod_dataset.xlsx"

        if self.selected_code == "en_transfer_learning.py":
          self.num_of_bases = 8
      elif self.transfer == "no":
        self.data = "datasets/izmir_iod_dataset.xlsx"
        self.hidden_size = 60
        self.model_count = 4

        if self.selected_code == "ENELM_Stacking.py":   
              self.hidden_size_final = 67
        elif self.selected_code == "ENELM_Voting.py":
              self.round_to = 0.02  
  
 


           
