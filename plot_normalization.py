"""
Generate horizontal bar chart of normalization types for the paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

labels = [
    'Punctuation\nRestoration',
    'Capitalization\nFixing',
    'Digit–Word\nCompounds',
    'All-Caps\nSegments',
    'Other',
]
sizes  = [84.9, 6.2, 4.1, 3.9, 0.9]
colors = ['#2166ac', '#4dac26', '#d7191c', '#fdae61', '#aaaaaa']

fig, ax = plt.subplots(figsize=(5.5, 2.8))

bars = ax.barh(labels[::-1], sizes[::-1], color=colors[::-1],
               height=0.55, edgecolor='white', linewidth=0.5)

# Value labels inside/outside bars
for bar, val in zip(bars, sizes[::-1]):
    x = bar.get_width()
    offset = 0.5 if x < 10 else -1.5
    ha = 'left' if x < 10 else 'right'
    color = 'black' if x < 10 else 'white'
    ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
            f'{val}%', va='center', ha=ha,
            fontsize=8.5, fontweight='bold', color=color)

ax.set_xlabel('Percentage of test examples (%)', fontsize=9)
ax.set_xlim(0, 96)
ax.set_title('Distribution of Normalization Types\n(Test Set, 1,000 examples)',
             fontsize=10, fontweight='bold', pad=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=8.5)
ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('./paper/latex/normalization_types.pdf',
            bbox_inches='tight', dpi=300)
plt.savefig('./paper/latex/normalization_types.png',
            bbox_inches='tight', dpi=300)
print("Saved normalization_types.pdf and .png")
