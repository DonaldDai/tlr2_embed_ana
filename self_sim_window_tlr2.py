import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA

def analyze_repeat_direct_window(embeddings, window_size=9, use_pca=False, pca_variance=0.95):
    """
    Perform sliding window smoothing on full dimensions.
    
    Args:
        embeddings: (L, D) matrix, e.g., (507, 1024)
        window_size: Window size (e.g., 9, 11, 15)
                     Larger windows make stripes more coherent but lower resolution.
        use_pca: Whether to use PCA for dimensionality reduction
        pca_variance: Variance ratio to retain in PCA (0.0 - 1.0)
    Returns:
        sim_matrix: (L, L) Self-similarity matrix
        smoothed_emb: (L, D) Smoothed feature matrix
    """
    # 1. Safety check
    if embeddings.ndim != 2:
        raise ValueError(f"Input must be a 2D matrix (L, D), current dimensions: {embeddings.ndim}")
    
    L, D = embeddings.shape
    print(f"Analyzing: Sequence Length={L}, Original Feature Dimension={D}, Window Size={window_size}")

    # 1.5 PCA Dimensionality Reduction (Optional)
    if use_pca:
        pca = PCA(n_components=pca_variance)
        embeddings = pca.fit_transform(embeddings)
        print(f"PCA Reduction: {D} -> {embeddings.shape[1]} (Retained {pca_variance*100}% Variance)")
        D = embeddings.shape[1]
    
    # 2. Full-Dimension Sliding Window Smoothing
    # I use convolution for moving average
    # mode='same' ensures output length matches input, boundaries are padded or handled automatically
    kernel = np.ones(window_size) / window_size
    
    # Convolve each column (dimension) separately
    # List comprehension is often faster and more readable than apply_along_axis here
    smoothed_emb = np.array([
        np.convolve(embeddings[:, d], kernel, mode='same') 
        for d in range(D)
    ]).T # Transpose back to (L, D)
    
    # 3. L2 Normalization (Critical Step)
    # ProtT5 vector magnitudes vary greatly, must normalize, otherwise dot product has no physical meaning
    norm = np.linalg.norm(smoothed_emb, axis=1, keepdims=True)
    # Add a small value to prevent division by zero
    normalized_emb = smoothed_emb / (norm + 1e-10)
    
    # 4. Calculate Self-Similarity Matrix
    # (L, D) @ (D, L) -> (L, L)
    sim_matrix = np.dot(normalized_emb, normalized_emb.T)
    
    return sim_matrix, normalized_emb

# ==========================================
# Actual Call Example: Compare Different Layers
# ==========================================

if __name__ == "__main__":
    for size in range(1, 12):
        # 1. Set Parameters
        # Target Protein UniProt ID (TLR2)
        UNIPROT_ID = 'O60603'
        # Sliding window size for smoothing
        WINDOW_SIZE = size
        # Whether to apply PCA dimensionality reduction
        USE_PCA = True
        # Variance ratio to retain in PCA
        PCA_VARIANCE = 0.95
        # Amino acid sequence range to analyze (start, end)
        RANGE = ((50, 550))
        layers = range(1, 25) 
        # Data directory (adjust according to your file structure)
        data_dir = './layer_embeds'
        
        print(f"Start analyzing self-similarity matrix for {UNIPROT_ID} across different layers (PCA={USE_PCA}, Var={PCA_VARIANCE})...")
        
        # Prepare canvas: 4 rows x 6 columns (total 24 plots)
        n_layers = len(layers)
        cols = 6
        rows = (n_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
        fig.suptitle(f"TLR2 UniProt ID: {UNIPROT_ID} Window Size: {WINDOW_SIZE}", fontsize=30)
        axes = axes.flatten()
        
        for i, layer in enumerate(layers):
            file_path = f"{data_dir}/TLR2_prot_t5_layer_{layer}.parquet"
            ax = axes[i]
            
            try:
                # Read Data
                print(f"Processing Layer {layer}...")
                data = pd.read_parquet(file_path)
                p_data = data[data['uniprot_id'] == UNIPROT_ID]
                
                if p_data.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_title(f"Layer {layer}")
                    continue
                    
                p_data = p_data.sort_values(by='position', ascending=True)
                raw_embeddings = np.array(p_data['embedding'][RANGE[0]:RANGE[1]].tolist())
                
                # Run Analysis
                sim_matrix, clean_feats = analyze_repeat_direct_window(
                    raw_embeddings, 
                    window_size=WINDOW_SIZE,
                    use_pca=USE_PCA,
                    pca_variance=PCA_VARIANCE
                )

                # Save similarity matrix
                # save_dir = './sim_matrices'
                # os.makedirs(save_dir, exist_ok=True)
                # sim_parquet_path = f"{save_dir}/TLR2_sim_matrix_win_size_{WINDOW_SIZE}_layer_{layer}.parquet"
                # pd.DataFrame(sim_matrix).to_parquet(sim_parquet_path)
                # print(f"Saved similarity matrix to {sim_parquet_path}")
                
                # Plotting
                im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f"Layer {layer}")
                ax.set_xlabel("Amino Acid Index", fontsize=8)
                ax.set_ylabel("Amino Acid Index", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
            except FileNotFoundError:
                ax.text(0.5, 0.5, 'File Not Found', ha='center', va='center')
                ax.set_title(f"Layer {layer}")
            except Exception as e:
                print(f"Error layer {layer}: {e}")
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.set_title(f"Layer {layer}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_file = f'TLR2_self_sim_layers_comparison_pca_{UNIPROT_ID}_window_{WINDOW_SIZE}.jpg'
        plt.savefig(output_file)
        print(f"Result saved to {output_file}")
