from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, sequence_data_handler, sequences):
        self.data_handler = sequence_data_handler
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        key, value = self.data_handler.extract_key_value(sequence)
        return key, value