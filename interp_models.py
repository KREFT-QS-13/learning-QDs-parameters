import os, json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import utilities.interpt_utils as iu
import utilities.model_utils as mu
from models.transfer_CNN import ResNet

def get_saliency_maps(model_name: str, mc_type: str, X, interpretor_model, save_images: bool = True, output_path: str = './Results_FINAL/Results/Interp_res'):
    gradcam_plus_plus = [] 
    guided_gradcam = [] 
    scorecam = [] 
    smoothgrad = [] 

    # Get the saliency maps for each image in the batch
    for i, x in enumerate(X):
        # gradcam_plus_plus.append(interpretor_model.compute_gradcam_plus_plus(x))
        guided_gradcam.append(interpretor_model.compute_guided_gradcam(x))
        scorecam.append(interpretor_model.compute_scorecam(x))
        # smoothgrad.append(interpretor_model.compute_smoothgrad(x))
        
        dict_masks_gcm = {
                    #   'Grad-CAM++':  gradcam_plus_plus[i], 
                      'Guided Grad-CAM': guided_gradcam[i],
                      'ScoreCAM': scorecam[i],
                    #   'SmoothGrad': smoothgrad[i]
                      }

        if save_images:
            if not os.path.exists(f'{output_path}/{model_name}/{mc_type}'):
                os.makedirs(f'{output_path}/{model_name}/{mc_type}')

            interpretor_model.visualize_all_masks(x,
                                                  dict_masks_gcm,
                                                  save_path=f'{output_path}/{model_name}/{mc_type}/gradcam_{model_name}_{mc_type}_{i}.png',
                                                  show=False)        
    return gradcam_plus_plus, guided_gradcam, scorecam, smoothgrad

def load_test_data(config_tuple, datasize_cut=1000):
    '''
    Load the test data
    '''
    X_gcm,y_gcm = mu.prepare_data(config_tuple, 
                            param_names=['csd', 'C_DD', 'C_DG'], 
                            all_batches=False,
                            batches=[33], 
                            datasize_cut=datasize_cut,
                            maxwell=True)

    X_lcm,y_lcm = mu.prepare_data(config_tuple, 
                            param_names=['csd', 'C_DD', 'C_DG'], 
                            all_batches=False,
                            batches=[33], 
                            datasize_cut=datasize_cut,
                            maxwell=False)

    return X_gcm, y_gcm, X_lcm, y_lcm

def load_model(model_name: str, mc_type: str, weights_path: str):
    '''
    Load the model from the weights path and check if the model is correct
    model_name: str, e.g. 'resnet18'
    mc_type: str, e.g. 'GCM' or 'LCM'
    weights_path: str, e.g. './Results_FINAL/Results/resnet18/Rn18-2-0_20250517_160844/weights.pth'
    '''

    if not model_name in weights_path:
        raise UserWarning(f'Model {model_name} not found in {weights_path}')
    
    with open(os.path.join(os.path.dirname(weights_path), 'results.json'), 'r') as f:
        results = json.load(f)
        mc_type_from_results = 'GCM' if results['param_names'][-1] == 'True' else 'LCM'
        if mc_type_from_results != mc_type:
            raise ValueError(f'Model {model_name} is {mc_type_from_results} but {mc_type} was specified')

        config_tuple = results['config']['params']['config_tuple']
        base_model = results['config']['params']['base_model']
        name = results['config']['params']['name']
        pretrained = results['config']['params']['pretrained']
        custom_head = results['config']['params']['custom_head']
        dropout = results['config']['params']['dropout']

    model = ResNet(config_tuple=config_tuple, 
                  name=name,
                  base_model=base_model,
                  pretrained=pretrained,
                  custom_head=custom_head, 
                  dropout=dropout)
    
    mu.load_model_weights(model, weights_path)

    print(f"Loaded models results from traning:")
    res = pd.read_csv(os.path.join(os.path.dirname(weights_path), 'results.csv'))
    print(f"{res}\n")

    return model

def evaluate_model(model, X, y):
    '''
    Evaluate the model on the test set
    '''
    model.eval()

    with torch.no_grad():
        pred = model(X)
    
    print("Obtained results on unsampled data:")
    MSE = mean_squared_error(pred, y)
    l2_norms = np.linalg.norm(y - pred, axis=1)

    print(f"MSE: {MSE}")
    print(f"L2 norms mean: {np.mean(l2_norms)}")

    return pred, l2_norms, MSE


def save_saliency_maps(model_name: str, mc_type: str, X, y, pred, l2_norms, gradcam_plus_plus, guided_gradcam, scorecam, smoothgrad, output_path: str = './Results_FINAL/Results/Interp_res/'):
    '''
    Save the saliency maps
    '''
    interp_dict = {'L2':  [x for x in l2_norms],
                        'y' : [x for x in y],
                        'y_pred' : [x for x in pred],
                        'X' : [x for x in X],
                        'gradcam_plus_plus' : [x for x in gradcam_plus_plus],
                        'guided_gradcam' : [x for x in guided_gradcam],
                        'scorecam' : [x for x in scorecam],
                        'smoothgrad' : [x for x in smoothgrad]}
    
    if not os.path.exists(f'{output_path}/{model_name}/{mc_type}'):
        os.makedirs(f'{output_path}/{model_name}/{mc_type}')
    np.savez_compressed(f"{output_path}/{model_name}/{mc_type}/interp_{model_name}_{mc_type}.npz", **interp_dict)  

def get_saliency_results(X, y, model_name: str, mc_type: str, weights_path: str, output_path: str = './Results_FINAL/Results/Interp_res/'):
    '''
    Get the saliency results
    '''
    model = load_model(model_name, mc_type, weights_path)

    pred, l2_norms, MSE = evaluate_model(model, X, y)

    interpretor_model = iu.InterpModel(model)
    gradcam_plus_plus, guided_gradcam, scorecam, smoothgrad = get_saliency_maps(model_name, mc_type, X, interpretor_model)

    save_saliency_maps(model_name, mc_type, X, y, pred, l2_norms, gradcam_plus_plus, guided_gradcam, scorecam, smoothgrad, output_path)

def plot_cluster_sizes(cluster_labels, cluster_stats, output_path=None, title='', datasize=1000):
    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors for GCM and LCM
    colors = {'GCM': '#2281c4', 'LCM': '#971c1c'}

    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_names = [cluster_stats[i]['sorted_label'] for i in unique]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Bar plot
    bars = ax.bar(cluster_names, counts, color=colors['GCM'])  # Using GCM color as default
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of elements')
    ax.set_title(title if title else 'Cluster Sizes')

    # Statistics box
    stat_lines = []
    for idx, stats in enumerate(cluster_stats):
        stat_lines.append(
            f"{stats['sorted_label']}: n={stats['size']}, "
            f"$\mu$={np.round(stats['mean_density'], 4) if stats['mean_density'] is not None else 'NA'} $\pm$ "
            f"{np.round(stats['std_density'], 4) if stats['std_density'] is not None else 'NA'}, "
        )
    stats_text = '\n'.join(stat_lines)
    # stats_text += '\n\nZero maps: ' + str(datasize - np.sum(counts))
    
    # Add text box with statistics
    fig.text(0.4, 0.76, stats_text,
            va='top', ha='left',
            fontsize=16,
            family='monospace',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='black',
                linewidth=1.5,
                alpha=1
            ))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_saliency_densities(saliency_maps, method_name: str, output_path: str = './Results_FINAL/Results/Interp_res/clustering', n_clusters_range: range = range(2, 11), title: str = '', return_clusters=False):
    """
    Analyze the density of active pixels in saliency maps using K-means clustering.
    
    Args:
        saliency_maps: List of saliency maps (numpy arrays)
        method_name: Name of the saliency method (e.g., 'Guided Grad-CAM' or 'ScoreCAM')
        output_path: Path to save the results and plots
        n_clusters_range: Range of number of clusters to try for elbow method
        title: Title for the plots
        return_clusters: Whether to return cluster labels, stats and optimal number of clusters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Calculate density of active pixels for each map
    densities = []
    for saliency_map in saliency_maps:
        active_pixels = np.sum(saliency_map > 0)
        total_pixels = saliency_map.size
        density = active_pixels / total_pixels
        densities.append(density)
    
    densities = np.array(densities).reshape(-1, 1)
    print(f"Non-zero maps: {len(densities)}, Zero maps: {len(saliency_maps) - len(densities)}.")
    
    # Calculate inertia and silhouette scores for different numbers of clusters
    inertias = []
    silhouette_scores = []
    k_values = []  # Store k values for which we calculate silhouette scores
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(densities)
        inertias.append(kmeans.inertia_)
        
        if n_clusters > 1:  # Silhouette score requires at least 2 clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(densities, labels))
            k_values.append(n_clusters)
    
    # Find optimal number of clusters using elbow method
    optimal_clusters = n_clusters_range[np.argmin(np.diff(inertias, 2)) + 1]
    
    # Perform final clustering with optimal number of clusters
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(densities)
    
    # Prepare cluster statistics (robust to empty clusters)
    cluster_stats = []
    for cluster in range(optimal_clusters):
        cluster_densities = densities[cluster_labels == cluster]
        if len(cluster_densities) == 0:
            stats = {
                'cluster': cluster + 1,
                'size': 0,
                'mean_density': None,
                'std_density': None,
                'min_density': None,
                'max_density': None
            }
        else:
            stats = {
                'cluster': cluster + 1,
                'size': len(cluster_densities),
                'mean_density': float(np.mean(cluster_densities)),
                'std_density': float(np.std(cluster_densities)),
                'min_density': float(np.min(cluster_densities)),
                'max_density': float(np.max(cluster_densities))
            }
        cluster_stats.append(stats)

    # --- Sort clusters by mean_density and relabel cluster_labels and stats ---
    # None values are treated as +inf so they go last
    cluster_stats_sorted = sorted(
        cluster_stats,
        key=lambda x: (float('inf') if x['mean_density'] is None else x['mean_density'])
    )
    # Assign new sorted labels
    for idx, stat in enumerate(cluster_stats_sorted):
        stat['sorted_label'] = f'C{idx+1}'
    # Create a mapping from old cluster index to new sorted index
    old_to_new = {}
    for new_idx, stat in enumerate(cluster_stats_sorted):
        old_to_new[stat['cluster'] - 1] = new_idx
    # Remap cluster_labels to sorted order
    cluster_labels_sorted = np.array([old_to_new[lab] for lab in cluster_labels])
    # -------------------------------------------------------------------------

    # Create visualization for elbow curve
    plt.figure(figsize=(6, 4))
    plt.plot(n_clusters_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal clusters: {optimal_clusters}')
    plt.legend()
    plt.tight_layout()
    
    # Save elbow curve plot
    elbow_plot_path = os.path.join(output_path, f"{method_name.replace(' ', '_')}_elbow_curve.png")
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create and save bar plot of cluster sizes with statistics box (sorted)
    bar_plot_path = os.path.join(output_path, f"{method_name.replace(' ', '_')}_cluster_sizes.png")
    plot_cluster_sizes(cluster_labels_sorted, cluster_stats_sorted, output_path=bar_plot_path, title=f'Cluster Sizes\n{title}')
    
    # Save sorted cluster statistics to CSV
    stats_df = pd.DataFrame(cluster_stats_sorted)
    stats_filename = f"{method_name.replace(' ', '_')}_cluster_stats.csv"
    stats_df.to_csv(os.path.join(output_path, stats_filename), index=False)
    
    # Print cluster statistics (sorted)
    print(f"\nCluster Analysis for {method_name}:")
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Results saved to: {output_path}")
    for stats in cluster_stats_sorted:
        print(f"\nCluster {stats['sorted_label']}:")
        print(f"Size: {stats['size']}")
        print(f"Mean density: {stats['mean_density']}")
        print(f"Std density: {stats['std_density']}")
        print(f"Min density: {stats['min_density']}")
        print(f"Max density: {stats['max_density']}")
    
    if return_clusters:
        return densities, cluster_labels_sorted, cluster_stats_sorted, optimal_clusters
    else:
        return densities, cluster_labels_sorted, optimal_clusters

def plot_cluster_sizes_comparison(gcm_cluster_labels, gcm_cluster_stats, lcm_cluster_labels, lcm_cluster_stats, 
                                output_path=None, title='', datasize=1000):
    """
    Create a grouped bar plot comparing GCM and LCM cluster sizes.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors for GCM and LCM
    colors = {'GCM': '#2281c4', 'LCM': '#971c1c'}

    # Get unique clusters and their counts for both GCM and LCM
    gcm_unique, gcm_counts = np.unique(gcm_cluster_labels, return_counts=True)
    lcm_unique, lcm_counts = np.unique(lcm_cluster_labels, return_counts=True)
    
    # Ensure we have the same clusters for both GCM and LCM
    all_clusters = np.unique(np.concatenate([gcm_unique, lcm_unique]))
    
    # Initialize counts arrays with zeros
    gcm_counts_full = np.zeros(len(all_clusters))
    lcm_counts_full = np.zeros(len(all_clusters))
    
    # Fill in the actual counts where we have data
    for i, cluster in enumerate(all_clusters):
        gcm_mask = gcm_unique == cluster
        lcm_mask = lcm_unique == cluster
        if np.any(gcm_mask):
            gcm_counts_full[i] = gcm_counts[gcm_mask][0]
        if np.any(lcm_mask):
            lcm_counts_full[i] = lcm_counts[lcm_mask][0]
    
    # Get cluster names from GCM stats (they should be the same for both)
    cluster_names = [gcm_cluster_stats[i]['sorted_label'] for i in range(len(all_clusters))]
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Set width of bars and positions of the bars
    barWidth = 0.35
    r1 = np.arange(len(cluster_names))
    r2 = [x + barWidth for x in r1]
    
    # Create grouped bar plot with specified colors
    bars1 = ax.bar(r1, gcm_counts_full, width=barWidth, color=colors['GCM'], label='GCM')
    bars2 = ax.bar(r2, lcm_counts_full, width=barWidth, color=colors['LCM'], label='LCM')
    
    # Add labels and title
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of elements')
    ax.set_title(title if title else 'Cluster Sizes Comparison (GCM vs LCM)')
    
    # Set x-axis ticks and labels
    ax.set_xticks([r + barWidth/2 for r in range(len(cluster_names))])
    ax.set_xticklabels(cluster_names)
    
    # Add legend
    ax.legend()
    
    # Add statistics text
    stat_lines = []
    for idx, (gcm_stats, lcm_stats) in enumerate(zip(gcm_cluster_stats, lcm_cluster_stats)):
        stat_lines.append(
            f"{gcm_stats['sorted_label']}: GCM(n={gcm_stats['size']}, "
            f"$\mu$={np.round(gcm_stats['mean_density'], 4) if gcm_stats['mean_density'] is not None else 'NA'}) "
            f"LCM(n={lcm_stats['size']}, "
            f"$\mu$={np.round(lcm_stats['mean_density'], 4) if lcm_stats['mean_density'] is not None else 'NA'})"
        )
    stats_text = '\n'.join(stat_lines)
    
    # Add text box with statistics
    fig.text(0.4, 0.76, stats_text,
            va='top', ha='left',
            fontsize=16,
            family='monospace',
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='black',
                linewidth=1.5,
                alpha=1
            ))
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_combined_saliency_densities(gcm_saliency_maps, lcm_saliency_maps, method_name: str, 
                                      output_path: str = './Results_FINAL/Results/Interp_res/clustering', 
                                      n_clusters_range: range = range(2, 11), title: str = ''):
    """
    Analyze the density of active pixels in saliency maps using K-means clustering on combined GCM and LCM data.
    
    Args:
        gcm_saliency_maps: List of GCM saliency maps
        lcm_saliency_maps: List of LCM saliency maps
        method_name: Name of the saliency method
        output_path: Path to save the results and plots
        n_clusters_range: Range of number of clusters to try
        title: Title for the plots
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Calculate densities for both GCM and LCM
    gcm_densities = []
    lcm_densities = []
    
    for saliency_map in gcm_saliency_maps:
        active_pixels = np.sum(saliency_map > 0)
        total_pixels = saliency_map.size
        density = active_pixels / total_pixels
        gcm_densities.append(density)
    
    for saliency_map in lcm_saliency_maps:
        active_pixels = np.sum(saliency_map > 0)
        total_pixels = saliency_map.size
        density = active_pixels / total_pixels
        lcm_densities.append(density)
    
    # Combine densities
    all_densities = np.array(gcm_densities + lcm_densities).reshape(-1, 1)
    print(f"Total non-zero maps: {len(all_densities)}, Total zero maps: {len(gcm_saliency_maps) + len(lcm_saliency_maps) - len(all_densities)}")
    
    # Calculate inertia and silhouette scores for different numbers of clusters
    inertias = []
    silhouette_scores = []
    k_values = []
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_densities)
        inertias.append(kmeans.inertia_)
        
        if n_clusters > 1:
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(all_densities, labels))
            k_values.append(n_clusters)
    
    # Find optimal number of clusters using elbow method
    optimal_clusters = n_clusters_range[np.argmin(np.diff(inertias, 2)) + 1]
    
    # Perform final clustering with optimal number of clusters
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    all_labels = final_kmeans.fit_predict(all_densities)
    
    # Split labels back into GCM and LCM
    gcm_labels = all_labels[:len(gcm_densities)]
    lcm_labels = all_labels[len(gcm_densities):]
    
    # Calculate cluster statistics for both GCM and LCM
    gcm_stats = []
    lcm_stats = []
    
    for cluster in range(optimal_clusters):
        # GCM stats
        gcm_cluster_densities = np.array(gcm_densities)[gcm_labels == cluster]
        gcm_stats.append({
            'cluster': cluster + 1,
            'size': len(gcm_cluster_densities),
            'mean_density': float(np.mean(gcm_cluster_densities)) if len(gcm_cluster_densities) > 0 else None,
            'std_density': float(np.std(gcm_cluster_densities)) if len(gcm_cluster_densities) > 0 else None,
            'min_density': float(np.min(gcm_cluster_densities)) if len(gcm_cluster_densities) > 0 else None,
            'max_density': float(np.max(gcm_cluster_densities)) if len(gcm_cluster_densities) > 0 else None
        })
        
        # LCM stats
        lcm_cluster_densities = np.array(lcm_densities)[lcm_labels == cluster]
        lcm_stats.append({
            'cluster': cluster + 1,
            'size': len(lcm_cluster_densities),
            'mean_density': float(np.mean(lcm_cluster_densities)) if len(lcm_cluster_densities) > 0 else None,
            'std_density': float(np.std(lcm_cluster_densities)) if len(lcm_cluster_densities) > 0 else None,
            'min_density': float(np.min(lcm_cluster_densities)) if len(lcm_cluster_densities) > 0 else None,
            'max_density': float(np.max(lcm_cluster_densities)) if len(lcm_cluster_densities) > 0 else None
        })
    
    # Sort clusters by mean density (using GCM means for consistency)
    mean_densities = [stats['mean_density'] if stats['mean_density'] is not None else float('inf') 
                     for stats in gcm_stats]
    sort_idx = np.argsort(mean_densities)
    
    # Sort both GCM and LCM stats
    gcm_stats_sorted = [gcm_stats[i] for i in sort_idx]
    lcm_stats_sorted = [lcm_stats[i] for i in sort_idx]
    
    # Assign new sorted labels
    for idx, (gcm_stat, lcm_stat) in enumerate(zip(gcm_stats_sorted, lcm_stats_sorted)):
        gcm_stat['sorted_label'] = f'C{idx+1}'
        lcm_stat['sorted_label'] = f'C{idx+1}'
    
    # Create mapping for new labels
    old_to_new = {old: new for new, old in enumerate(sort_idx)}
    
    # Remap labels to sorted order
    gcm_labels_sorted = np.array([old_to_new[lab] for lab in gcm_labels])
    lcm_labels_sorted = np.array([old_to_new[lab] for lab in lcm_labels])
    
    return (gcm_labels_sorted, gcm_stats_sorted, 
            lcm_labels_sorted, lcm_stats_sorted, 
            optimal_clusters)

def main():
    config_tuple = (3,2,1)
    datasize_cut = 1000
    X_gcm, y_gcm, X_lcm, y_lcm = load_test_data(config_tuple, datasize_cut)

    # Paths to the weights
    LCM_weights_path_10 = './Results_FINAL/Results/resnet10/Rn10-2-1_20250517_182042/Rn10-2-1-.pth'
    GCM_weights_path_10 = './Results_FINAL/Results/resnet10/Rn10-2-1_20250517_172057/Rn10-2-1-.pth'

    LCM_weights_path_18 = './Results_FINAL/Results/resnet18/Rn18-2-1_20250517_180722/Rn18-2-1-.pth'
    GCM_weights_path_18 = './Results_FINAL/Results/resnet18/Rn18-2-1_20250517_174156/Rn18-2-1-.pth'

    # print("Getting saliency results for GCM 10")
    # get_saliency_results(X_gcm, y_gcm, 'resnet10', 'GCM', GCM_weights_path_10)
    # print("Done")
    
    # print("Getting saliency results for LCM 10")
    # get_saliency_results(X_lcm, y_lcm, 'resnet10', 'LCM', LCM_weights_path_10)
    # print("Done")

    # print("Getting saliency results for GCM 18")
    # get_saliency_results(X_gcm, y_gcm, 'resnet18', 'GCM', GCM_weights_path_18)
    # print("Done")

    # print("Getting saliency results for LCM 18")
    # get_saliency_results(X_lcm, y_lcm, 'resnet18', 'LCM', LCM_weights_path_18)
    # print("Done")


    # Load and Analyze saliency densities
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet10/GCM/interp_resnet10_GCM.npz')['guided_gradcam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-gcm',
        title='Guided Grad-CAM (Resnet10 GCM)'  
    )
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet10/GCM/interp_resnet10_GCM.npz')['scorecam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-gcm',
        title='ScoreCAM (Resnet10 GCM)'
    )

    # LCM
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet10/LCM/interp_resnet10_LCM.npz')['guided_gradcam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-lcm',
        title='Guided Grad-CAM (Resnet10 LCM)'
    )
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet10/LCM/interp_resnet10_LCM.npz')['scorecam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-lcm',
        title='ScoreCAM (Resnet10 LCM)'
    )


    # GCM   
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet18/GCM/interp_resnet18_GCM.npz')['guided_gradcam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-gcm',
        title='Guided Grad-CAM (Resnet18 GCM)'
    )
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet18/GCM/interp_resnet18_GCM.npz')['scorecam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-gcm',
        title='ScoreCAM (Resnet18 GCM)'
    )
    # LCM
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet18/LCM/interp_resnet18_LCM.npz')['guided_gradcam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-lcm',
        title='Guided Grad-CAM (Resnet18 LCM)'
    )
    saliency_maps = np.load(f'./Results_FINAL/Results/Interp_res/resnet18/LCM/interp_resnet18_LCM.npz')['scorecam']
    analyze_saliency_densities(
        saliency_maps=saliency_maps,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-lcm',
        title='ScoreCAM (Resnet18 LCM)'
    )

    # Load GCM and LCM results for the same architecture
    gcm_results = np.load('./Results_FINAL/Results/Interp_res/resnet10/GCM/interp_resnet10_GCM.npz')
    lcm_results = np.load('./Results_FINAL/Results/Interp_res/resnet10/LCM/interp_resnet10_LCM.npz')
    
    # Get saliency maps
    gcm_ggcam = gcm_results['guided_gradcam']
    lcm_ggcam = lcm_results['guided_gradcam']
    gcm_scorecam = gcm_results['scorecam']
    lcm_scorecam = lcm_results['scorecam']
    
    # Analyze combined GCM and LCM data
    gcm_labels, gcm_stats, lcm_labels, lcm_stats, optimal_clusters = analyze_combined_saliency_densities(
        gcm_saliency_maps=gcm_ggcam,
        lcm_saliency_maps=lcm_ggcam,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-combined',
        title='Guided Grad-CAM (Resnet10 GCM vs LCM)'
    )
    
    # Create comparison plot
    plot_cluster_sizes_comparison(
        gcm_labels, gcm_stats,
        lcm_labels, lcm_stats,
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-comparison.png',
        title='Cluster Sizes Comparison (Resnet10 GCM vs LCM) - Guided Grad-CAM'
    )

    gcm_labels, gcm_stats, lcm_labels, lcm_stats, optimal_clusters = analyze_combined_saliency_densities(
        gcm_saliency_maps=gcm_scorecam,
        lcm_saliency_maps=lcm_scorecam,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-combined',
        title='ScoreCAM (Resnet10 GCM vs LCM)'
    )    

    plot_cluster_sizes_comparison(
        gcm_labels, gcm_stats,
        lcm_labels, lcm_stats,
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn10-comparison-scorecam.png',
        title='Cluster Sizes Comparison (Resnet10 GCM vs LCM) - ScoreCAM'
    )


        # Load GCM and LCM results for the same architecture
    gcm_results = np.load('./Results_FINAL/Results/Interp_res/resnet18/GCM/interp_resnet18_GCM.npz')
    lcm_results = np.load('./Results_FINAL/Results/Interp_res/resnet18/LCM/interp_resnet18_LCM.npz')
    
    # Get saliency maps
    gcm_ggcam = gcm_results['guided_gradcam']
    lcm_ggcam = lcm_results['guided_gradcam']
    gcm_scorecam = gcm_results['scorecam']
    lcm_scorecam = lcm_results['scorecam']
    
    # Analyze combined GCM and LCM data
    gcm_labels, gcm_stats, lcm_labels, lcm_stats, optimal_clusters = analyze_combined_saliency_densities(
        gcm_saliency_maps=gcm_ggcam,
        lcm_saliency_maps=lcm_ggcam,
        method_name='Guided Grad-CAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-combined',
        title='Guided Grad-CAM (Resnet18 GCM vs LCM)'
    )
    
    # Create comparison plot
    plot_cluster_sizes_comparison(
        gcm_labels, gcm_stats,
        lcm_labels, lcm_stats,
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-comparison.png',
        title='Cluster Sizes Comparison (Resnet18 GCM vs LCM) - Guided Grad-CAM'
    )

    gcm_labels, gcm_stats, lcm_labels, lcm_stats, optimal_clusters = analyze_combined_saliency_densities(
        gcm_saliency_maps=gcm_scorecam,
        lcm_saliency_maps=lcm_scorecam,
        method_name='ScoreCAM',
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-combined',
        title='ScoreCAM (Resnet18 GCM vs LCM)'
    )    

    plot_cluster_sizes_comparison(
        gcm_labels, gcm_stats,
        lcm_labels, lcm_stats,
        output_path='./Results_FINAL/Results/Interp_res/clustering-rn18-comparison-scorecam.png',
        title='Cluster Sizes Comparison (Resnet18 GCM vs LCM) - ScoreCAM'
    )

if __name__ == '__main__':
    main()

