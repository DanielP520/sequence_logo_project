# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:18:58 2023

@author: camlo
"""
import os.path
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import glob
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import math
import logomaker
import numpy as np
import helper_functions


def create_3d_graph(df1, df2):
    # Get XYZ positions from the DataFrame columns
    x1, y1, z1 = df1['X'], df1['Y'], df1['Z']
    x2, y2, z2 = df2['X'], df2['Y'], df2['Z']

    color_shapely = df2['shapely'].values.tolist()
    color_polar = df2['polar'].values.tolist()
    init_notebook_mode(connected=True)
    # Create traces for the scatter plots
    scatter_trace1 = go.Scatter3d(
        x=x1,
        y=y1,
        z=z1,
        mode='markers',
        marker=dict(
            size=9,
            color=color_shapely,
            opacity=0
        ),
        uid='trace1'
    )

    scatter_trace2 = go.Scatter3d(
        x=x2,
        y=y2,
        z=z2,
        mode='markers',
        marker=dict(
            size=6,
            color='black',
            opacity=0
        ),
        name='Target',
        uid='trace2'
    )
    buttons = []
    buttons.append(dict(label='color shapely', method='restyle',  args=[{'marker.color': [color_shapely]}, [0]]))
    buttons.append(dict(label='Polar', method='restyle', args=[{'marker.color': [color_polar]}, [0]]))
    updatemenus = [
        dict(buttons=buttons, showactive=True),
        dict(direction='down', x=0.1, xanchor='left', y=1.1, yanchor='top'),
    ]

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            camera=dict(
                eye=dict(x=2, y=-2, z=1.5)  # Adjust the eye position to view all eight regions
            )
        ),
        title='Interactive 3D Scatter Plot'
    )

    # Create the figure and add the traces
    fig = go.Figure(data=[scatter_trace1, scatter_trace2], layout=layout)
    fig.update_layout(updatemenus=updatemenus)
    # Show the interactive plot
    iplot(fig)


def create_3d_graph_list(df, df_list, sequence_info):
    # Create a list to store the scatter traces
    scatter_traces = []

    x, y, z = df['X'], df['Y'], df['Z']
    scatter_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=15,
            color=1,
            opacity=0.5
        ),
        hoverinfo='text',
        hovertext=f'TARGET',

    )
    scatter_traces.append(scatter_trace)

    for i, df in enumerate(df_list):

        x, y, z = df['X'], df['Y'], df['Z']
        target_postions = list(map(str, list(set(list(df['target_residues'].values)))))
        target_str = " ".join(target_postions)
        seq_info_per_df = {col: value for col, value in sequence_info[i].iloc[0].items() if value != 0}
        seq_str = ""
        for col, value in seq_info_per_df.items():
            seq_str += (f"{col}: {value}" + '<br>')

        scatter_trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=10,
                color=i + 2,
                opacity=0
            ),
            hoverinfo='text',
            hovertext=f'Target Residues : {target_str}, & Index: {i + 1} <br> {seq_str}',
            name=f'Index: {i + 1}'
        )

        # Add the scatter trace to the list
        scatter_traces.append(scatter_trace)

    # Create the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            camera=dict(
                eye=dict(x=2, y=-2, z=1.5)  # Adjust the eye position to view all eight regions
            )
        ),
        title='Interactive 3D Scatter Plot',
        showlegend=False
    )

    # Create the figure and add the traces
    fig = go.Figure(data=scatter_traces, layout=layout)

    # Show the interactive plot
    fig.show(renderer='browser')


def find_nearest_points(target, binders, radius):
    # Convert XYZ positions of target DataFrame to a list of tuples
    target_points = list(target[['X', 'Y', 'Z']].values)
    positions = list(target['residue_index'].values)

    # Convert XYZ positions of binders DataFrame to a list of tuples
    binder_points = list(binders[['X', 'Y', 'Z']].values)

    # Create KDTree for binder_points
    tree = KDTree(binder_points)

    nearest_points_indices = []
    for i, target_point in enumerate(target_points):
        # Query binders DataFrame within the specified radius of target_point
        indices = tree.query_ball_point(target_point, r=radius)
        if not indices:
            continue
        nearest_points_indices.append([indices, positions[i]])

    # Create a new DataFrame with nearest points
    nearest_points_df = pd.DataFrame()
    for indices in nearest_points_indices:
        temp_df = binders.iloc[indices[0]]
        nearest_points_df = pd.concat([nearest_points_df, temp_df], ignore_index=True)

    return nearest_points_df


def cluster_3d_positions(df, num_clusters):
    # Extract the 3D positions from the DataFrame
    positions = df[['X', 'Y', 'Z']].values

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(positions)

    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Create a list of DataFrames for each cluster
    clusters = []
    for cluster_id in range(num_clusters):
        cluster_df = df[cluster_labels == cluster_id]
        clusters.append(cluster_df)

    # Return the list of DataFrames
    return clusters


def transform_to_1_letter_code(amino_acids_3_letter):
    # Mapping dictionary for 3-letter to 1-letter code
    aa_mapping = {
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'CYS': 'C',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'ILE': 'I',
        'LEU': 'L',
        'LYS': 'K',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V'
    }

    amino_acids_1_letter = [aa_mapping[aa] for aa in amino_acids_3_letter]
    return amino_acids_1_letter


def calculate_frequency(character, lst):
    count = lst.count(character)

    frequency = count / len(lst)

    return frequency


def calculate_bits(list_of_AA, sequence_list):
    list_of_frequencies = []
    for AA in list_of_AA:
        list_of_frequencies.append(calculate_frequency(AA, sequence_list))
    S = math.log2(20)
    H = 0

    for f in list_of_frequencies:
        if f == 0:
            continue
        H = H + (-f) * (math.log2(f))
    H = H + 19 / (np.log(2) * 2 * len(sequence_list))
    R = S - H

    heights = []
    for f in list_of_frequencies:
        heights.append(np.abs(f * 100))
    return heights


def create_sequence_logo(df, target, logos_made):
    # Create a Logo object

    logo = logomaker.Logo(df,
                          color_scheme='NajafabadiEtAl2017')
    logo.ax.set_ylabel('bits')
    positions = [i for i in range(len(target))]
    logo.ax.set_xticklabels(target)
    logo.ax.set_xticks(positions)
    plt.savefig(f'logo_{logos_made}.png')


def create_sequence_logo_list(df_list, target):
    # Create a Logo object
    for i, df in enumerate(df_list):
        logo = logomaker.Logo(df,
                              color_scheme='NajafabadiEtAl2017'
                              )
        logo.ax.set_ylabel('percentange')

        positions = [i for i in range(len(target[i]))]
        logo.ax.set_aspect('auto')

        logo.ax.set_xticklabels([target[i]])
        logo.ax.set_xticks([1])
        name = os.path.join("temp_seq_logos", str(i + 1) + ".png")
        plt.savefig(name)


def plot_sequence_logo(df, filename=None):
    # Calculate the height of each letter for each position
    stacked_df = df.apply(lambda row: pd.Series(row.sort_values(ascending=False).values), axis=1)
    bottom = stacked_df.cumsum(axis=1).shift(1, axis=1).fillna(0)
    top = bottom + df.values

    # Plotting the sequence logo
    fig, ax = plt.subplots(figsize=(10, 5))

    for base in df.columns:
        ax.fill_between(df.index, bottom[base], top[base], alpha=0.8, linewidth=0.4)

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Position')
    ax.set_ylabel('Probability')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure as an image file
    else:
        plt.show()


def plot(files, target_chain, binder, is_ligand):
    list_of_paths = glob.glob(files + "/*.pdb")
    target_chain_cordinates = helper_functions.extract_info_pdb(list_of_paths[0], target_chain)
    data_frame_target = pd.DataFrame(target_chain_cordinates)

    binder_chain_cordinates = []
    for file in list_of_paths:
        binder_chain_cordinates += helper_functions.extract_info_pdb(file,binder)
    data_frame_binders = pd.DataFrame(binder_chain_cordinates)
    nearest_neighbors_df = find_nearest_points(data_frame_target, data_frame_binders, 7)
    create_3d_graph(nearest_neighbors_df,data_frame_target)


    exit(1)


    cluster_sequences = []
    list_of_target_positions = []

    for cluster in clustered_list:
        list_of_target_positions.append(list(set(list(cluster['target_residues'].values))))
        cluster_sequences.append(transform_to_1_letter_code(cluster['AA'].values.tolist()))

    model = logomaker.get_example_matrix('ww_information_matrix',
                                         print_description=False)
    list_of_AA = model.columns.to_list()
    list_of_data_frames = []
    for i in cluster_sequences:
        bits = calculate_bits(list_of_AA, i)
        df = pd.DataFrame(columns=model.columns)
        df.loc[0] = bits
        list_of_data_frames.append(df)

    create_3d_graph_list(data_frame_target, clustered_list, list_of_data_frames)
    another_logo = 1
    logos_made = 1
    # while another_logo:
    #     selected_clusters = input("Enter the cluster index (comma-separated) you want to make a sequence logo for: ")
    #     selected_clusters = [int(label.strip()) - 1 for label in selected_clusters.split(',')]
    #     list_selected_clusters = []
    #     list_of_targets = []
    #     for i in selected_clusters:
    #         list_of_targets.append(list(set(list(clustered_list[i]['target_residues'].values))))
    #         list_selected_clusters.append(transform_to_1_letter_code(clustered_list[i]['AA'].values.tolist()))
    #     model = logomaker.get_example_matrix('ww_information_matrix',
    #                                          print_description=False)
    #     list_of_AA = model.columns.to_list()
    #     bits_selected_clusters = []
    #     for i in list_selected_clusters:
    #         bits_selected_clusters.append(calculate_bits(list_of_AA, i))
    #     df = pd.DataFrame(columns=model.columns)
    #     df = pd.concat([df, pd.DataFrame(bits_selected_clusters, columns=df.columns)], ignore_index=True)
    #     create_sequence_logo(df, list_of_targets, logos_made)
    #     another_logo = input("Would you like to make another logo? 0 for no, 1 for yes.")
    #     logos_made +=1
