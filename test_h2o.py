import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Initialize H2O
h2o.init()

# Create sample dataset
data = {
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    'target': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# Convert to H2O Frame
h2o_df = h2o.H2OFrame(df)

# Split data
train, test = h2o_df.split_frame(ratios=[0.8])

# Run AutoML
aml = H2OAutoML(max_models=5, seed=1)
aml.train(x=['feature1', 'feature2'], y='target', training_frame=train)

# View leaderboard
print("AutoML Leaderboard:")
print(aml.leaderboard.head())

# Shutdown H2O
h2o.shutdown(prompt=False)
print("H2O installation test completed successfully!")