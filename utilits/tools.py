from plotly import graph_objects as go
import plotly.io as pio

import os
import sys
import numpy as np
import pandas as pd
from numpy import random as npr

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams['text.usetex'] = True



def group_points_by_minimum_error(points_df) -> pd.DataFrame:
    groupped = points_df[["a1", "a2", "rmse", "sample_size"]].groupby(
        by=["a1", "a2", "sample_size"], as_index=False
    ).min().join(
        points_df.set_index(["a1", "a2", "rmse", "sample_size"]), 
        on=["a1", "a2", "rmse", "sample_size"]
    )
    
    renamed = groupped.rename(
        columns={
            "a1": "α₁",
            "a2": "α₂",
            "rmse": "RMSE",
            "model_name": "Model"
        }
    )

    renamed["Model"] = renamed["Model"].map(
    {
        "knn": "kNN",
        "svd": "SVD",
        "nmf": "NMF",
        "autorec": 'AutoRec'
    })
    
    return renamed




def visualize_3d_plot(results, sample_sizes, height=800, width=1000, dot_size=3, printer=True, save_path=None):    
    name_to_color = {
        "kNN": "yellow",
        "SVD": "red",
        "NMF": "blue",
        "AutoRec": "green"
    }
    for sample_size in sample_sizes:
        data = results[results["sample_size"] == sample_size]
        fig = go.Figure(
            layout=go.Layout(
                height=height,
                width=width,
                font=dict(size=16),
                margin=dict(l=20, r=20, t=20, b=20),
                scene=dict(
                    xaxis = dict(title={"text": "α₂" , "font": {"size": 30}}, tickfont = {"size": 20}),
                    yaxis = dict(title={"text": "α₁", "font": {"size": 30}}, tickfont = {"size": 20}, 
                                 tickvals = [0.2, 0.4, 0.6, 0.8, 1]),
                    zaxis = dict(title={"text": " " , "font": {"size": 30}}, tickfont = {"size": 20}, rangemode='tozero'),
                    aspectmode='cube'
                ),
                scene_camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.2, y=1.8, z=1.0)
                ),
            ),
            data=[
                go.Scatter3d(
                    name=model_name,
                    x=data[data["Model"] == model_name]["α₂"],
                    y=data[data["Model"] == model_name]["α₁"],
                    z=data[data["Model"] == model_name]["RMSE"],
                    mode="markers",
                    marker=dict(
                        size=dot_size,
                        color=name_to_color[model_name],
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                ) for model_name in ["NMF", "SVD", "kNN", 'AutoRec']
            ]
        )
        fig.update_layout(showlegend=False)
        if save_path != None:
            pio.write_image(fig, save_path, height=height, width=width)
        if printer:
            fig.show("notebook")
            

def visualize_2d_plot(data, sample_size, legend='upper right', save_path = None):
    data_sample = data[(data.a1 == 0.2) & (data.sample_size == sample_size)]
    
    nmf_mean = data_sample[data_sample['model_name'] == 'nmf'].groupby(['a2'])['rmse'].mean()
    svd_mean = data_sample[data_sample['model_name'] == 'svd'].groupby(['a2'])['rmse'].mean()
    knn_mean = data_sample[data_sample['model_name'] == 'knn'].groupby(['a2'])['rmse'].mean()
    autorec_mean = data_sample[data_sample['model_name'] == 'autorec'].groupby(['a2'])['rmse'].mean()

    nmf_std = data_sample[data_sample['model_name'] == 'nmf'].groupby(['a2'])['rmse'].std()
    svd_std = data_sample[data_sample['model_name'] == 'svd'].groupby(['a2'])['rmse'].std()
    knn_std = data_sample[data_sample['model_name'] == 'knn'].groupby(['a2'])['rmse'].std()
    autorec_std = data_sample[data_sample['model_name'] == 'autorec'].groupby(['a2'])['rmse'].std()
    
    ax1 = plt.subplot()

    ax1.plot(np.arange(0, 0.9, 0.1), list(nmf_mean), color='blue', label='NMF')
    ax1.fill_between(np.arange(0, 0.9, 0.1), [el1-el2 for (el1, el2) in zip(list(nmf_mean),list(nmf_std))], 
                [el1+el2 for (el1, el2) in zip(list(nmf_mean),list(nmf_std))], color='blue', alpha=.2)

    ax1.plot(np.arange(0, 0.9, 0.1), list(svd_mean), color='red', label='SVD')
    ax1.fill_between(np.arange(0, 0.9, 0.1), [el1-el2 for (el1, el2) in zip(list(svd_mean),list(svd_std))], 
                [el1+el2 for (el1, el2) in zip(list(svd_mean),list(svd_std))], color='red', alpha=.2)


    ax1.plot(np.arange(0, 0.9, 0.1), list(knn_mean), color='yellow', label='kNN')
    ax1.fill_between(np.arange(0, 0.9, 0.1), [el1-el2 for (el1, el2) in zip(list(knn_mean),list(knn_std))], 
                [el1+el2 for (el1, el2) in zip(list(knn_mean),list(knn_std))], color='yellow', alpha=.2)

    ax1.plot(np.arange(0, 0.9, 0.1), list(autorec_mean), color='green', label='AutoRec')
    ax1.fill_between(np.arange(0, 0.9, 0.1), [el1-el2 for (el1, el2) in zip(list(autorec_mean),list(autorec_std))], 
                [el1+el2 for (el1, el2) in zip(list(autorec_mean),list(autorec_std))], color='green', alpha=.2)

    ax1.legend(loc=legend, prop={'size': 20})
    ax1.set_xlabel(r'$\alpha_2$', fontsize=27)
    ax1.set_ylabel(r'$RMSE$', fontsize=27)
    
    ax1.tick_params(axis="x", labelsize=25)
    ax1.tick_params(axis="y", labelsize=25)
    
    if save_path != None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches = 0.1)
        
        

        
        
def boxplot(data1, data2, column='rmse', hue='model_name', legend_loc='upper right', save_path=None):
    
    comp_data_real = comparison_table(data1, data2).sort_values(by=['sample_size', 'rmse_mean_real', 'rmse_std_real'])
    comp_data_syn = comparison_table(data1, data2).sort_values(by=['sample_size', 'rmse_mean_syn', 'rmse_std_syn'])
    
    #hue_names = sorted(data1[hue].unique())[::-1]
    hue_names_real = list(comp_data_real[hue])
    hue_names_syn = list(comp_data_syn[hue])
    
    colors_dict = {'nmf':'blue', 'svd':'red', 'knn':'yellow', 'autorec':'green'}
    colors_whiskers_dict = {'blue':'#CCCCFF', 'red':'#FFCCCC', 'yellow':'#FFFFCC', 'green':'#CCE5CC'}
    
    colors_list_real=list(map(colors_dict.get, hue_names_real))
    colors_list_syn=list(map(colors_dict.get, hue_names_syn))
    
    colors = []
    for color in colors_list_real:
        colors.append(color)
        
    for color in colors_list_syn:
        colors.append(color)
        
    box_data = []
    for name in hue_names_real:
        box_data.append(data1[data1[hue]==name][column])
    
    for name in hue_names_syn:
        box_data.append(data2[data2[hue]==name][column])
    
    
    ax1 = plt.subplot()
    bp = ax1.boxplot(box_data, patch_artist = True, vert = 1)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # changing color and linewidth of whiskers
    index=0
    for whisker in bp['whiskers']:
        i=index//2
        whisker.set(color = colors_whiskers_dict[colors[i]], linewidth = 5, linestyle =":")
        index+=1
    
    # changing color and linewidth of caps
    index=0
    for cap in bp['caps']:
        i=index//2
        cap.set(color =colors[i], linewidth = 3)
        index+=1
        
    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='black', linewidth = 3)
        
    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='+', color ='red', alpha = 0.3)

    ax1.tick_params(axis="y", labelsize=22)
    ax1.tick_params(axis="x", labelsize=22, length=0)
    plt.xticks([2.5, 6.5], [r'real data', r'synthetic data'], rotation ='horizontal', fontsize=25)
    
    legend_labels = list(map({'nmf':'NMF', 'svd':'SVD', 'knn':'kNN', 'autorec':'AutoRec'}.get, hue_names_real))
    legend_labels_ordered = list(map({'NMF':0, 'SVD':1, 'kNN':2, 'AutoRec':3}.get, legend_labels))
    boxes_ordered = [x for _, x in sorted(zip(legend_labels_ordered, [bp["boxes"][i] for i in range(0, len(box_data)//2, 1)]))]
    labels_ordered = [x for _, x in sorted(zip(legend_labels_ordered, legend_labels))]
    ax1.legend(boxes_ordered, labels_ordered, loc=legend_loc, fontsize=20)
    

    if save_path != None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches = 0.1)
    plt.show()
    
    
    
    
    
    
    
    
def comparison_table(data1, data2):
    real_grouped_mean=data1.groupby(by=['sample_size', 'model_name'], as_index=False)[['rmse', 'mae']].mean().sort_values(by=['sample_size', 'rmse', 'mae']).reset_index(drop=True)
    real_grouped_std=data1.groupby(by=['sample_size', 'model_name'], as_index=False)[['rmse', 'mae']].std().sort_values(by=['sample_size', 'rmse', 'mae']).reset_index(drop=True)
    real_grouped_mean=real_grouped_mean.rename(columns={'rmse':'rmse_mean', 'mae':'mae_mean'})
    real_grouped_std=real_grouped_std.rename(columns={'rmse':'rmse_std', 'mae':'mae_std'})
    real_stat=pd.merge(real_grouped_mean, real_grouped_std, on=['sample_size', 'model_name'])
    
    syn_grouped_mean = data2.groupby(by = ['sample_size', 'model_name'], as_index=False)[['rmse', 'mae']].mean().sort_values(by=['sample_size', 'rmse', 'mae']).reset_index(drop=True)
    syn_grouped_std = data2.groupby(by = ['sample_size', 'model_name'], as_index=False)[['rmse', 'mae']].std().sort_values(by=['sample_size', 'rmse', 'mae']).reset_index(drop=True)
    syn_grouped_mean=syn_grouped_mean.rename(columns={'rmse':'rmse_mean', 'mae':'mae_mean'})
    syn_grouped_std=syn_grouped_std.rename(columns={'rmse':'rmse_std', 'mae':'mae_std'})
    syn_stat=pd.merge(syn_grouped_mean, syn_grouped_std, on=['sample_size', 'model_name'])
    
    stat = pd.merge(real_stat, syn_stat, on=['sample_size', 'model_name']).rename(
    columns={'rmse_mean_x':'rmse_mean_real',
            'mae_mean_x':'mae_mean_real',
            'rmse_std_x':'rmse_std_real',
            'mae_std_x':'mae_std_real',
            'rmse_mean_y':'rmse_mean_syn',
            'mae_mean_y':'mae_mean_syn',
            'rmse_std_y':'rmse_std_syn',
            'mae_std_y':'mae_std_syn',
            }
    )
    
    return stat