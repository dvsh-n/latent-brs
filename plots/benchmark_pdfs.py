import matplotlib.pyplot as plt
import numpy as np

# Tasks and Methods
tasks = ['Reacher', 'Cube', '2-arm Rope']
methods = ['Ours', 'LeWM', 'DINO-WM', 'PLDM']

# Success Data
success_means = {
    'Reacher': [83.5, 17.0, 24.5, 40.5],
    'Cube': [91.5, 28.5, 69.0, 28.75],
    '2-arm Rope': [93.75, 13.0, 25.0, 7.0]
}
success_stds = {
    'Reacher': [4.5, 7.6, 6.2, 9.9],
    'Cube': [4.2, 3.8, 2.95, 6.62],
    '2-arm Rope': [2.71, 4.3, 7.91, 5.34]
}

# Distance Data
dist_means = {
    'Reacher': [0.3769, 1.3052, 0.5332, 0.8207],
    'Cube': [0.0506, 0.1331, 0.0539, 0.1449],
    '2-arm Rope': [0.02025, 0.0930, 0.0578, 0.1303]
}
dist_stds = {
    'Reacher': [0.7870, 1.1091, 0.1150, 0.1583],
    'Cube': [0.0551, 0.0227, 0.0483, 0.0051],
    '2-arm Rope': [0.00231, 0.0146, 0.0016, 0.0190]
}

colors = ['#a1d99b', '#f59c9d', '#aec7e8', '#ffbb78'] 
edgecolors = ['#74c476', '#e64b4c', '#5a9bd4', '#f16913']

def plot_bar(ax, task, means_dict, stds_dict, ylabel, ylim, is_int_label=True):
    means = means_dict[task]
    stds = stds_dict[task]
    
    ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor=edgecolors, 
           linewidth=1.5, error_kw=dict(lw=1.5, capthick=1.5))
    
    ax.set_title(task, fontsize=14, pad=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim(0, ylim)
    
    for i, v in enumerate(means):
        txt = f"{v:.1f}" if is_int_label else f"{v:.3f}"
        ax.text(i, max(v/2, ylim*0.05), txt, ha='center', va='center', fontweight='bold', color='black', fontsize=10)
        
    ax.tick_params(axis='x', rotation=30, labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

# -----------------------------------------
# Plot 1: Success Rate
# -----------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
for i, task in enumerate(tasks):
    ylabel = r'Success Rate (%) ($\uparrow$)' if i == 0 else ''
    plot_bar(axes1[i], task, success_means, success_stds, ylabel, 100, is_int_label=True)
plt.tight_layout()
plt.savefig('plots/success_rate.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

# -----------------------------------------
# Plot 2: Minimum Distance
# -----------------------------------------
dist_ylims = {
    'Reacher': 2.5,
    'Cube': 0.25,
    '2-arm Rope': 0.2
}
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
for i, task in enumerate(tasks):
    ylabel = r'Min Distance ($\downarrow$)' if i == 0 else ''
    plot_bar(axes2[i], task, dist_means, dist_stds, ylabel, dist_ylims[task], is_int_label=False)
plt.tight_layout()
plt.savefig('plots/min_distance.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)