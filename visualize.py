import torch
import matplotlib.pyplot as plt
from enum import Enum

class AP_Metric(Enum):
    PROBABILITY = "probability"
    RANK = "rank"

def plot_token_metrics_line(
    source_logits,
    target_logits_clean,
    target_logits_patched,
    tokenizer,
    metric = AP_Metric.RANK,
    token_ids = None,
):
    """
    Create a line plot showing the probability of token(s) across layers.
    """

    if token_ids is None:
        token_ids = []
        token_ids.append(target_logits_clean.argmax(dim=-1).item())
        token_ids.append(source_logits.argmax(dim=-1).item())

    # Convert single token to list for uniform handling
    if not isinstance(token_ids, list):
        token_ids = [token_ids]
    
    layers = sorted([int(k) for k in target_logits_patched.keys()])
    
    plt.figure(figsize=(12, 6))
    
    # Calculate clean probabilities
    clean_probs = torch.softmax(target_logits_clean, dim=-1)
    
    # Plot a line for each token
    for token_idx in token_ids:
        token_metrics = list()
        
        for layer in layers:
            logits = target_logits_patched[str(layer)]
            if metric == AP_Metric.PROBABILITY:
                probs = torch.softmax(logits, dim=-1)
                prob_at_idx = probs[token_idx].item()
                token_metrics.append(prob_at_idx)
            elif metric == AP_Metric.RANK:
                sorted_indices = torch.argsort(logits, descending=True)
                rank_at_idx = (sorted_indices == token_idx).nonzero(as_tuple=True)[0].item() + 1  
                token_metrics.append(rank_at_idx)
        
        # Create label with token name if tokenizer provided
        if tokenizer is not None:
            token_name = tokenizer.decode([token_idx])
            label = f'{token_name} (idx: {token_idx})'
            clean_label = f'{token_name} (clean)'
        else:
            label = f'Token {token_idx}'
            clean_label = f'Token {token_idx} (clean)'
        
        # Plot patched probabilities
        line = plt.plot(layers, token_metrics, marker='o', linewidth=2, markersize=6, label=label)[0]
        
        # Add horizontal dashed line for clean probability with same color
        if metric == AP_Metric.PROBABILITY:
            clean_metric = clean_probs[token_idx].item()
        elif metric == AP_Metric.RANK:
            sorted_indices = torch.argsort(target_logits_clean, descending=True)
            clean_metric = (sorted_indices == token_idx).nonzero(as_tuple=True)[0].item() + 1  
        plt.axhline(y=clean_metric, linestyle='--', linewidth=2, alpha=0.7, color=line.get_color(), label=clean_label)
    
    plt.xlabel('Layer Patched', fontsize=18)

    if metric == AP_Metric.RANK:
        plt.ylabel('Rank', fontsize=18)
        plt.autoscale(axis='y', tight=False)
        plt.ylim(bottom=0.5)
        yticks = list(plt.yticks()[0])
        if 1 not in yticks:
            yticks.append(1)
        plt.yticks(sorted(yticks))
        plt.gca().invert_yaxis()
    elif metric == AP_Metric.PROBABILITY:
        plt.ylabel('Probability', fontsize=18)  
        plt.ylim(0, 1)
    
    plt.title('Token Statistic Evolution', fontsize=20)
    plt.legend(fontsize=11)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()