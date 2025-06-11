import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

output_dir = os.path.join('output')
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")


def plot_ngrams(ngram_df: pd.DataFrame, title: str, output_path: str):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Frequency', y='Ngram', data=ngram_df, hue='type' if 'type' in ngram_df else None)
    plt.title(title, fontsize=20)
    plt.xlabel("Frequency", fontsize=20)
    plt.ylabel("")
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, fontsize=20)  # Rotate y-ticks and adjust fontsize
    plt.tight_layout()
    plt.savefig(output_path)
    
    
def create_thank_you_hist(df: pd.DataFrame):
    
    thank_you_count = df[df.text.eq(' Thank you.')] \
        .groupby('movie_name') \
        .text.count() \
        .reset_index() \
        .rename(columns={'text': 'count'})
        
    missed_movies = set(df.movie_name.unique()).difference(set(thank_you_count['movie_name']))
    missed_rows = []

    for movie in missed_movies:
        missed_rows.append({'movie_name': movie, 'count': 0})
        
    thank_you_count = pd.concat([pd.DataFrame(missed_rows), thank_you_count])

    sns.histplot(thank_you_count, bins=10, legend=False)
    plt.title('Histogram of " Thank you." Hallucinations by Movie')
