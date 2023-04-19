import pandas as pd

# Load the data into a pandas DataFrame
data = pd.read_csv('./online_shoppers_intention.csv')

# Split the data into positive and negative samples based on the values in the 'column_name' column
positive_samples = data[data['Revenue'] == True]
negative_samples = data[data['Revenue'] == False]

# Save the positive and negative samples to separate Excel sheets
with pd.ExcelWriter('./sampled_data.xlsx') as writer:
    positive_samples.to_excel(writer, sheet_name='Positive Samples', index=False)
    negative_samples.to_excel(writer, sheet_name='Negative Samples', index=False)