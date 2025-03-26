import pickle 

with open('data/hits.pkl', "rb") as f:
    hits = pickle.load(f)

print('tmem72_chr10' in hits)