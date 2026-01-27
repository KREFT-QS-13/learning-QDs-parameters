import argparse
import torch
import numpy as np
import time
import json

import src.utilities.model_utils as mu
import src.utilities.utils as u
from src.models.multi_branch import MultiBranchArchitecture

def tsem(model_config_path:str, num_dps:int=None):
    '''
    Train and evaluate models on the given dataset.
    Args:
        model_config_path: Path to the model configuration file.
        num_dps: Number of datapoints to use. If -1, use all datapoints.
    '''

    # Load configuration from JSON file
    try:
        with open(model_config_path, 'r') as f:
            confs = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {model_config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {model_config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file {model_config_path}: {e}")
    
    # Extract paths
    if 'paths' not in confs:
        raise KeyError("'paths' key not found in configuration file")
    dataset_path = confs['paths']['dataset_path']
    main_dir_results = confs['paths']['main_dir_results']
    
    print(f"Loading dataset from {dataset_path}...")
    data, imgs, _, _ = u.load_all_data(dataset_path, load_images=True) # imgs already preprocessed
    context = u.create_context(data)
    outputs = u.create_outputs(data)
    assert len(imgs) == len(context) == len(outputs), "Data alignment error: imgs={len(imgs)}, context={len(context)}, outputs={len(outputs)}"

    if num_dps is not None and num_dps>0 and num_dps<=len(context):
        context = context[:num_dps]
        outputs = outputs[:num_dps]
        imgs = imgs[:num_dps]
        print(f"Data loaded. Using {num_dps} datapoints from {dataset_path}.")
    elif num_dps is None:
        print(f"Data loaded. Using all {len(context)} datapoints from {dataset_path}.")
    else:
        raise ValueError(f"Number of datapoints to use must be between 0 and {len(context)}. Got {num_dps}.")
    
    # TODO prepare to be passed to the model in 

    # High level model configuration
    model_name = confs['model']['name']
    num_branches = confs['model']['num_branches']
    dropout = confs['model']['dropout']
    output_size = confs['model']['output_size']
    
    # Image encoder configuration
    image_encoder = confs['model']['image_encoder']
    branch_model = image_encoder['type']
    pretrained = image_encoder['pretrained']
    filters_per_layer = image_encoder['filters_per_layer']
    custom_cnn_layers = image_encoder['custom_cnn_layers']
    
    # Context encoder configuration
    context_encoder = confs['model']['context_encoder']
    context_vector_size = context_encoder['context_vector_size']
    context_embedding_dim = context_encoder['context_embedding_dim']
    context_hidden_dims = context_encoder.get('hidden_dims', None)
    
    # Branch fusion configuration
    branch_fusion = confs['model']['branch_fusion']
    branch_predicition_head = branch_fusion['hidden_dims']
    branch_embedding_dim = branch_fusion['branch_embedding_dim']
    
    # Attention configuration
    attention = confs['model']['attention']
    pooling_method = attention['method']
    num_attention_heads = attention['num_heads']
    
    # Final prediction head configuration
    # Check if separate configurations for diagonal and off-diagonal heads are provided
    if 'diagonal_head' in confs['model'] and 'off_diagonal_head' in confs['model']:
        # Separate configurations for each head
        diagonal_head_config = confs['model']['diagonal_head'].get('hidden_dims', None)
        off_diagonal_head_config = confs['model']['off_diagonal_head'].get('hidden_dims', None)
        final_prediction_head = None  # Not used when separate configs are provided
        print(f"Using separate configurations for diagonal and off-diagonal heads.")
    else:
        # Fallback to shared configuration
        final_prediction_head = confs['model']['final_prediction_head']['hidden_dims']
        diagonal_head_config = None
        off_diagonal_head_config = None
        print(f"Using shared configuration for both prediction heads.")
    
    print(f"Training model {model_name} with {num_branches} branches, {branch_model} as image encoder, and {pooling_method} attention method.")

    # Check available devices and print type and device name
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"CUDA is available. Number of GPU devices: {num_devices}")
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU device {i}: {device_name}")
        print(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        print(f"CUDA is not available. Using CPU.")
        print(f"Device: cpu")
    
    model = MultiBranchArchitecture(name=model_name,
                                    branch_model=branch_model,
                                    pooling_method=pooling_method,
                                    num_branches=num_branches,
                                    custom_cnn_layers=custom_cnn_layers,
                                    pretrained=pretrained,
                                    filters_per_layer=filters_per_layer,
                                    dropout=dropout,
                                    branch_predicition_head=branch_predicition_head,
                                    context_vector_size=context_vector_size,
                                    num_attention_heads=num_attention_heads,
                                    branch_embedding_dim=branch_embedding_dim,
                                    final_prediction_head=final_prediction_head,
                                    diagonal_head_config=diagonal_head_config,
                                    off_diagonal_head_config=off_diagonal_head_config,
                                    output_size=output_size,
                                    context_embedding_dim=context_embedding_dim,
                                    context_hidden_dims=context_hidden_dims)
    
    print(f"""Detailed model parameters: 
    - name: {model_name}
    - num_branches: {num_branches}
    - dropout: {dropout}
    - output_size: {output_size}
    
    Image Encoder:
    - type: {branch_model}
    - pretrained: {pretrained}
    - filters_per_layer: {filters_per_layer}
    - custom_cnn_layers: {custom_cnn_layers}
    
    Context Encoder:
    - context_vector_size: {context_vector_size}
    - context_embedding_dim: {context_embedding_dim}
    - hidden_dims: {context_hidden_dims}
    
    Branch Fusion:
    - hidden_dims: {branch_predicition_head}
    - branch_embedding_dim: {branch_embedding_dim}
    
    Attention:
    - method: {pooling_method}
    - num_heads: {num_attention_heads}
    
    Final Prediction Head:
    - shared hidden_dims: {final_prediction_head if final_prediction_head is not None else 'N/A (using separate configs)'}
    - diagonal_head hidden_dims: {diagonal_head_config if diagonal_head_config is not None else 'N/A (using shared config)'}
    - off_diagonal_head hidden_dims: {off_diagonal_head_config if off_diagonal_head_config is not None else 'N/A (using shared config)'}""")

    # Count parameters per branch
    print("\nModel parameters breakdown:")
    branch_params = []
    for i, branch in enumerate(model.branches):
        branch_total = sum(p.numel() for p in branch.parameters())
        branch_trainable = sum(p.numel() for p in branch.parameters() if p.requires_grad)
        branch_params.append(branch_total)
        print(f"  Branch {i+1}: {branch_total:,} (Trainable: {branch_trainable:,})")
    
    # Count attention module parameters
    if hasattr(model, 'attention'):
        attn_total = sum(p.numel() for p in model.attention.parameters())
        attn_trainable = sum(p.numel() for p in model.attention.parameters() if p.requires_grad)
        print(f"  Attention module: {attn_total:,} (Trainable: {attn_trainable:,})")
    
    # Count final prediction head parameters (diagonal + off-diagonal heads)
    if hasattr(model, 'diagonal_head') and hasattr(model, 'off_diagonal_head'):
        diag_head_total = sum(p.numel() for p in model.diagonal_head.parameters())
        diag_head_trainable = sum(p.numel() for p in model.diagonal_head.parameters() if p.requires_grad)
        off_diag_head_total = sum(p.numel() for p in model.off_diagonal_head.parameters())
        off_diag_head_trainable = sum(p.numel() for p in model.off_diagonal_head.parameters() if p.requires_grad)
        head_total = diag_head_total + off_diag_head_total
        head_trainable = diag_head_trainable + off_diag_head_trainable
        print(f"  Final prediction heads (diagonal + off-diagonal): {head_total:,} (Trainable: {head_trainable:,})")
        print(f"    - Diagonal head: {diag_head_total:,} (Trainable: {diag_head_trainable:,})")
        print(f"    - Off-diagonal head: {off_diag_head_total:,} (Trainable: {off_diag_head_trainable:,})")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total model parameters: {total_params:,} (Trainable: {trainable_params:,})\n")
    
    batch_size = confs['train']['batch_size']
    epochs = confs['train']['epochs']
    learning_rate = confs['train']['learning_rate']
    val_split = confs['train']['val_split']
    test_split = confs['train']['test_split']
    random_state = confs['train']['random_state']
    epsilon = confs['train']['epsilon']
    # New loss function parameters
    reg_coeff_diag = confs['train'].get('reg_coeff_diag', 0.1)
    reg_coeff_off = confs['train'].get('reg_coeff_off', 1.0)

    use_scaling = confs['train']['use_scaling']
    use_lr_scheduler = confs['train']['use_lr_scheduler']
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results = mu.train_evaluate_and_save_models(
        model=model,
        imgs=imgs,
        context=context,
        outputs=outputs,
        save_dir=main_dir_results,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        val_split=val_split,
        test_split=test_split,
        random_state=random_state,
        epsilon=epsilon,
        reg_coeff_diag=reg_coeff_diag,
        reg_coeff_off=reg_coeff_off,
        use_scaling=use_scaling,
        use_lr_scheduler=use_lr_scheduler
    )

    print("Training, evaluation, and saving complete!")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument('-c', type=str, required=False, default='configs/model/model_config.json',   
                        help='Path to the configuration file to the whole process of training, evaluating and saving the model. Mandatory. Format: json.')
    parser.add_argument('-n', type=int, required=False, default=None, 
                        help='Number of datapoints to use. If None, use all datapoints.')
    
    args = parser.parse_args()
    tsem(args.c, args.n)

if __name__ == "__main__":
    main()
