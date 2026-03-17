import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('../GIST_Train.csv')
top1 = 'PREDICT_original_tf_GLCM_energyd3.0A0.0'
top2 = 'PREDICT_original_tf_GLCM_energyd3.0A2.36'

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x=top1, y=top2, hue='label', s=80, alpha=0.7)
plt.title(f'Top 2 Features Dot Plot\\n{top1} vs {top2} (colored by label)')
plt.xlabel(top1)
plt.ylabel(top2)
plt.legend(title='Label')
plt.tight_layout()
plt.savefig('gist_top2_scatter.png', dpi=200, bbox_inches='tight')
plt.show()
print('Saved gist_top2_scatter.png and shown plot.')
