import pandas as pd
from sklearn.model_selection import train_test_split
from load_data import load_data

print("Loading GIST radiomic features data...")
df = load_data()
print(f"Original data shape: {df.shape}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Split into train/test (75/25 stratified)
train_df, test_df = train_test_split(
    df, 
    test_size=0.25, 
    random_state=42, 
    stratify=df['label']
)

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Train class distribution:\n{train_df['label'].value_counts()}")
print(f"Test class distribution:\n{test_df['label'].value_counts()}")

# Save the splits
train_df.to_csv('GIST_Train.csv', index=False)
test_df.to_csv('GIST_test.csv', index=False)

print("\nFiles saved:")
print("- GIST_Train.csv")
print("- GIST_test.csv")
print("\nSplit complete!")

