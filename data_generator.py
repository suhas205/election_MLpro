import numpy as np
import pandas as pd

def generate_sample_data(n_samples=1000):
    """Generate sample election data."""
    np.random.seed(42)
    
    # Generate features
    age_group = np.random.choice(['18-25', '26-35', '36-50', '51+'], n_samples)
    income_level = np.random.normal(50000, 20000, n_samples)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    urban_rural = np.random.choice(['Urban', 'Rural'], n_samples)
    previous_voting = np.random.choice(['Yes', 'No'], n_samples)
    
    # Create some correlation between features and outcome
    win_probability = (
        (age_group == '51+').astype(float) * 0.3 +
        (income_level > 50000).astype(float) * 0.2 +
        (np.isin(education_level, ['Master', 'PhD'])).astype(float) * 0.2 +
        (urban_rural == 'Urban').astype(float) * 0.15 +
        (previous_voting == 'Yes').astype(float) * 0.15 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Generate election outcome (1 for win, 0 for loss)
    outcome = (win_probability > 0.5).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age_group': age_group,
        'income_level': income_level,
        'education_level': education_level,
        'urban_rural': urban_rural,
        'previous_voting': previous_voting,
        'outcome': outcome
    })
    
    return data

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_sample_data()
    data.to_csv('election_data.csv', index=False)
    print("Sample election data generated and saved to 'election_data.csv'")
