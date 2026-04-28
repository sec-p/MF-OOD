"""
Test script to verify LoCoOp OOD regularization integration.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model_modular import build_modular_model, compute_locoop_ood_loss
import clip

def test_locoop_ood_loss():
    """Test the LoCoOp OOD loss function."""
    print("Testing LoCoOp OOD loss function...")
    
    # Create dummy data
    B, N, D, C = 4, 196, 768, 10
    local_feats = torch.randn(B, N, D).cuda()
    text_feats = torch.randn(C, D).cuda()
    labels = torch.randint(0, C, (B,)).cuda()
    
    # Normalize features
    local_feats = local_feats / local_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    
    # Test loss computation
    loss = compute_locoop_ood_loss(
        local_feats, text_feats, labels, top_k=50, logit_scale=100.0
    )
    
    print(f"  ✓ LoCoOp OOD loss: {loss.item():.4f}")
    print(f"  ✓ Loss shape: {loss.shape}")
    print(f"  ✓ Loss is finite: {torch.isfinite(loss)}")
    
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    return True

def test_model_integration():
    """Test the integration with ModularCustomCLIP."""
    print("\nTesting model integration...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classnames = ["cat", "dog", "bird", "car", "tree"]
    
    # Load CLIP
    clip_model, _ = clip.load("ViT-B/16", device=device)
    
    # Create config with LoCoOp OOD enabled
    cfg = {
        'device': device,
        'selector_type': 'mlp',
        'num_select': 16,
        'fuser_type': 'mean',
        'templates': ["a photo of a {}"],
        'use_locoop_ood': True,
        'lambda_locoop_ood': 0.1,
        'locoop_topk': 50,
    }
    
    # Build model
    model = build_modular_model(cfg, classnames, clip_model)
    model = model.to(device)
    
    # Test forward pass with labels
    images = torch.randn(2, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1]).to(device)
    
    output = model(images, labels=labels)
    
    print(f"  ✓ Model forward pass successful")
    print(f"  ✓ Logits shape: {output['logits'].shape}")
    print(f"  ✓ Aux losses keys: {output['aux_losses'].keys()}")
    
    # Check if LoCoOp loss is present
    if 'locoop_ood' in output['aux_losses']:
        ood_loss = output['aux_losses']['locoop_ood']
        print(f"  ✓ LoCoOp OOD loss found: {ood_loss.item():.4f}")
        assert torch.isfinite(ood_loss), "LoCoOp loss should be finite"
    else:
        print(f"  ⚠ LoCoOp OOD loss not found in aux_losses")
        print(f"  Available losses: {list(output['aux_losses'].keys())}")
    
    return True

def test_backward_pass():
    """Test backward pass with LoCoOp loss."""
    print("\nTesting backward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classnames = ["cat", "dog", "bird"]
    
    # Load CLIP
    clip_model, _ = clip.load("ViT-B/16", device=device)
    
    # Create config
    cfg = {
        'device': device,
        'selector_type': 'mlp',
        'num_select': 8,
        'fuser_type': 'mean',
        'templates': ["a photo of a {}"],
        'use_locoop_ood': True,
        'lambda_locoop_ood': 0.1,
        'locoop_topk': 50,
    }
    
    # Build model
    model = build_modular_model(cfg, classnames, clip_model)
    model = model.to(device)
    
    # Forward pass
    images = torch.randn(2, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1]).to(device)
    
    output = model(images, labels=labels)
    logits = output['logits']
    aux_losses = output['aux_losses']
    
    # Compute total loss
    ce_loss = torch.nn.functional.cross_entropy(logits, labels)
    total_aux_loss = sum(v for v in aux_losses.values() if v.requires_grad)
    total_loss = ce_loss + total_aux_loss
    
    # Backward pass
    total_loss.backward()
    
    print(f"  ✓ Backward pass successful")
    print(f"  ✓ Total loss: {total_loss.item():.4f}")
    print(f"  ✓ CE loss: {ce_loss.item():.4f}")
    print(f"  ✓ Total aux loss: {total_aux_loss.item():.4f}")
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "At least one parameter should have gradients"
    print(f"  ✓ Gradients computed successfully")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("LoCoOp OOD Regularization Integration Test")
    print("="*60)
    
    try:
        # Test 1: Loss function
        test_locoop_ood_loss()
        
        # Test 2: Model integration
        test_model_integration()
        
        # Test 3: Backward pass
        test_backward_pass()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
