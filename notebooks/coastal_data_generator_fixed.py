import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class RealisticCoastalDataGenerator:
    """
    Generate realistic coastal sensor data based on oceanographic research patterns
    Memory-optimized version
    """
    def __init__(self, sequence_length=168, n_samples=500):  # 168 hours = 1 week, reduced from 3000 to 500
        self.sequence_length = sequence_length
        self.n_samples = n_samples
        
    def generate_coastal_dataset(self):
        """Generate realistic coastal monitoring time series data"""
        np.random.seed(42)
        
        # Time-based features
        time_points = np.arange(self.n_samples * self.sequence_length)
        hours = time_points % 24
        days = (time_points // 24) % 365
        months = (days // 30.44) % 12
        
        # Tidal patterns (semi-diurnal tides, ~12.4 hour period)
        tidal_component = 2.0 * np.sin(2 * np.pi * time_points / 12.4) + \
                         0.5 * np.sin(2 * np.pi * time_points / 24.8)
        
        # Seasonal patterns
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * days / 365.25 - np.pi/2)
        seasonal_salinity = 35 + 2 * np.sin(2 * np.pi * days / 365.25)
        
        # Storm events (random extreme events)
        storm_events = np.random.exponential(0.02, len(time_points))
        storm_mask = storm_events > np.percentile(storm_events, 95)
        
        # Generate sensor readings
        data = {
            # Water quality parameters
            'pH': 8.1 + 0.3 * np.sin(2 * np.pi * days / 365) + \
                  0.1 * tidal_component + np.random.normal(0, 0.05, len(time_points)),
            
            'salinity': seasonal_salinity + 0.5 * tidal_component + \
                       np.random.normal(0, 0.2, len(time_points)),
            
            'dissolved_oxygen': 8.5 + 1.5 * np.sin(2 * np.pi * days / 365) - \
                               0.3 * (seasonal_temp - 15) / 10 + \
                               np.random.normal(0, 0.3, len(time_points)),
            
            'temperature': seasonal_temp + 2 * np.sin(2 * np.pi * hours / 24) + \
                          np.random.normal(0, 0.5, len(time_points)),
            
            'turbidity': 5 + 3 * tidal_component + 10 * storm_mask.astype(float) + \
                        np.random.exponential(1, len(time_points)),
            
            # Physical parameters
            'wave_height': 1.2 + 0.8 * np.sin(2 * np.pi * days / 365) + \
                          2.0 * storm_mask.astype(float) + \
                          np.random.exponential(0.3, len(time_points)),
            
            'current_speed': 0.3 + 0.2 * np.abs(tidal_component) + \
                           0.5 * storm_mask.astype(float) + \
                           np.random.exponential(0.1, len(time_points)),
            
            'water_level': tidal_component + 0.2 * np.sin(2 * np.pi * days / 365) + \
                          np.random.normal(0, 0.1, len(time_points)),
            
            # Meteorological
            'air_pressure': 1013 + 15 * np.sin(2 * np.pi * days / 365) - \
                           20 * storm_mask.astype(float) + \
                           np.random.normal(0, 3, len(time_points)),
            
            'wind_speed': 5 + 3 * np.sin(2 * np.pi * days / 365) + \
                         15 * storm_mask.astype(float) + \
                         np.random.exponential(2, len(time_points)),
            
            # Time features
            'hour_sin': np.sin(2 * np.pi * hours / 24),
            'hour_cos': np.cos(2 * np.pi * hours / 24),
            'day_sin': np.sin(2 * np.pi * days / 365),
            'day_cos': np.cos(2 * np.pi * days / 365),
        }
        
        # Create erosion risk based on multiple factors
        erosion_score = (
            0.3 * (data['wave_height'] - np.mean(data['wave_height'])) / np.std(data['wave_height']) +
            0.2 * (data['current_speed'] - np.mean(data['current_speed'])) / np.std(data['current_speed']) +
            0.2 * (data['wind_speed'] - np.mean(data['wind_speed'])) / np.std(data['wind_speed']) +
            0.1 * (data['turbidity'] - np.mean(data['turbidity'])) / np.std(data['turbidity']) +
            0.1 * storm_mask.astype(float) +
            0.1 * np.random.normal(0, 1, len(time_points))
        )
        
        # Convert to categorical erosion risk
        erosion_risk = np.zeros(len(time_points), dtype=int)
        erosion_risk[erosion_score > np.percentile(erosion_score, 80)] = 2  # High risk
        erosion_risk[(erosion_score > np.percentile(erosion_score, 50)) & 
                    (erosion_score <= np.percentile(erosion_score, 80))] = 1  # Medium risk
        # Low risk = 0 (default)
        
        data['erosion_risk'] = erosion_risk
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='H')
        
        return df
    
    def create_sequences(self, df, target_col='erosion_risk', max_sequences=5000, stride=2):
        """
        Create overlapping sequences for time series modeling with memory constraints
        
        Parameters:
        - df: DataFrame with sensor data
        - target_col: Target column name
        - max_sequences: Maximum number of sequences to create
        - stride: Step size between sequences to reduce memory usage
        
        Returns:
        - sequences: numpy array of input sequences
        - targets: numpy array of target values
        - feature_cols: list of feature column names
        """
        feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
        
        # Use a batch approach to avoid loading everything into memory at once
        sequences = []
        targets = []
        
        # Use stride to skip some sequences and reduce memory usage
        for i in range(0, min(len(df) - self.sequence_length, max_sequences*stride), stride):
            seq = df[feature_cols].iloc[i:i+self.sequence_length].values
            target = df[target_col].iloc[i+self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
            # Check if we've reached our limit
            if len(sequences) >= max_sequences:
                break
                
        # Use float32 instead of float64 to reduce memory usage
        return np.array(sequences, dtype=np.float32), np.array(targets), feature_cols

class CoastalDataset(Dataset):
    """PyTorch Dataset for coastal sensor data"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# Example usage:
if __name__ == "__main__":
    print("Generating realistic coastal sensor dataset...")
    generator = RealisticCoastalDataGenerator(sequence_length=168, n_samples=500)
    coastal_df = generator.generate_coastal_dataset()

    print(f"Generated dataset shape: {coastal_df.shape}")
    print(f"Features: {list(coastal_df.columns)}")
    print(f"Erosion risk distribution: {coastal_df['erosion_risk'].value_counts().sort_index()}")

    # Create sequences for modeling with memory constraints
    sequences, targets, feature_names = generator.create_sequences(
        coastal_df, max_sequences=5000, stride=3
    )
    print(f"\nSequence data shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Feature count: {len(feature_names)}")
    print(f"Memory usage of sequences: {sequences.nbytes / (1024 * 1024):.2f} MB") 