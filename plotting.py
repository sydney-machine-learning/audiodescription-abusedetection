import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

output_dir = os.path.join('output')
os.makedirs(output_dir, exist_ok=True)

sns.set_palette('colorblind')

# sns.set_style('whitegrid')

# The following settings were identified with assistance from ChatGPT, then tweaked based on experimentation
plt.rcParams.update({
    'figure.figsize': (3.5, 2.5),
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.title_fontsize': 7,
    'lines.linewidth': 1,
    'lines.markersize': 4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})


acb_palette = ['#fff100', '#00adee', '#ED1C24']
full_acb_palette = ['#afafaf', '#0db14b', '#fff100', '#00adee', '#ED1C24', '#000000']


def savefig(filename: str):
    plt.savefig(os.path.join(output_dir, filename + '.png'), bbox_inches='tight')


def place_legend_ontop_of_graph(g, is_catplot: bool, legend_title: str, ncol: int, txt_labels=None):

    if is_catplot:
        g._legend.remove()
        handles, labels = g.ax.get_legend_handles_labels()
    else:
        g.legend().remove()
        handles, labels = g.get_legend_handles_labels()

    labels = txt_labels if txt_labels is not None else labels

    legend = g.figure.legend(
        handles=handles,
        labels=labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.07 + 0.05 * (len(labels) / ncol)),
        ncol=ncol,
        frameon=False,
        fontsize=7,
        title=legend_title,
        title_fontsize=7
    )
    plt.tight_layout()


def plot_ngrams(ngram_df: pd.DataFrame, title: str, output_path: str):
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Percentage', y='Ngram', data=ngram_df, hue='Type' if 'Type' in ngram_df else None)
    # plt.title(title, fontsize=20)
    plt.xlabel('Percentage (%)', fontsize=20)
    plt.ylabel('')
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(rotation=45, fontsize=16)  # Rotate y-ticks and adjust fontsize
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    
    
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
