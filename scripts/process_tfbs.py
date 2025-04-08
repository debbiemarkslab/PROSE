import pandas as pd

top = set(pd.read_table('/n/groups/marks/users/courtney/projects/regulatory_genomics/models/LOL-EVE_private/benchmarks/tfbs_removal/notebooks/top_1_genes.txt', header=None)[0])
bottom = set(pd.read_table('/n/groups/marks/users/courtney/projects/regulatory_genomics/models/LOL-EVE_private/benchmarks/tfbs_removal/notebooks/bottom_1_genes.txt', header=None)[0])

df = pd.read_csv('/n/groups/marks/users/erik/Promoter_Poet_private/data/tfbs.csv')

result_df = df.copy()
    
# Initialize the expression column with None
result_df['expression'] = None

# Set values based on the lists
result_df.loc[result_df['GENE'].isin(top), 'expression'] = 'variable'
result_df.loc[result_df['GENE'].isin(bottom), 'expression'] = 'consistent'
result_df = result_df[result_df['expression'].notna()]

result_df.to_csv('/n/groups/marks/users/erik/Promoter_Poet_private/model/tfbs_with_expression_types.csv')
