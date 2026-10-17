import pytest
import torch
import torch.nn.functional as F

from robustdrive.foundation_jepa import (
    FoundationCorruptionJEPA,
    MaskedPredictor,
    ema_momentum,
    select_tokens,
)
from robustdrive.foundation_v2 import FoundationViTDetectorV2


def test_select_tokens():
    tokens = torch.randn(2, 10, 32)
    indices = torch.tensor([[0, 2, 4], [1, 3, 5]])
    selected = select_tokens(tokens, indices)
    assert selected.shape == (2, 3, 32)
    assert torch.allclose(selected[0, 0], tokens[0, 0])
    assert torch.allclose(selected[0, 1], tokens[0, 2])
    assert torch.allclose(selected[1, 2], tokens[1, 5])


def test_ema_momentum():
    assert ema_momentum(0.996, 0, 1000) == pytest.approx(0.996, abs=1e-5)
    assert ema_momentum(0.996, 1000, 1000) == pytest.approx(1.0, abs=1e-5)
    mid = ema_momentum(0.996, 500, 1000)
    assert 0.996 < mid < 1.0


def test_masked_predictor_forward():
    predictor = MaskedPredictor(count=960, encoder_dim=384, dim=192, depth=2, heads=4)
    context = torch.randn(2, 480, 384)
    visible = torch.arange(0, 480).unsqueeze(0).expand(2, -1)
    target_indices = torch.arange(480, 960).unsqueeze(0).expand(2, -1)
    out = predictor(context, visible, target_indices)
    assert out.shape == (2, 480, 384)


def test_foundation_corruption_jepa_forward_backward():
    # Test with pretrained=False for fast lightweight unit test
    model = FoundationCorruptionJEPA(
        model_name="vit_small_patch16_384",
        img_size=(384, 640),
        pretrained=False,
        mask_ratio=0.50,
        predictor_depth=2,
        predictor_heads=4,
    )
    model.train()
    clean = torch.randn(2, 3, 384, 640)
    view = torch.randn(2, 3, 384, 640)

    loss, metrics = model(clean, view)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    assert "prediction_loss" in metrics
    assert "variance_loss" in metrics
    assert "covariance_loss" in metrics

    loss.backward()
    # Check that online parameters have gradients and target parameters DO NOT
    for p in model.online.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break

    for p in model.target.parameters():
        assert p.grad is None

    # Test EMA update after modifying online parameters
    with torch.no_grad():
        for p in model.online.parameters():
            p.add_(torch.ones_like(p))
            break
    initial_target_p = next(model.target.parameters()).clone()
    model.update_target(0.996)
    updated_target_p = next(model.target.parameters()).clone()
    assert not torch.allclose(initial_target_p, updated_target_p)


def test_student_encoder_transfer_to_detector():
    jepa = FoundationCorruptionJEPA(
        model_name="vit_small_patch16_384",
        img_size=(384, 640),
        pretrained=False,
    )
    detector = FoundationViTDetectorV2(
        model_name="vit_small_patch16_384",
        img_size=(384, 640),
        pretrained=False,
    )

    # Load JEPA student weights into detector backbone
    detector.backbone.load_state_dict(jepa.get_student_encoder().state_dict())

    # Verify detector forward pass works with transferred backbone
    dummy_input = torch.randn(2, 3, 384, 640)
    preds = detector(dummy_input)
    assert len(preds) == 3
    # P3: (B, 1, 48, 80, 15), P4: (B, 1, 24, 40, 15), P5: (B, 1, 12, 20, 15)
    assert preds[0].shape == (2, 1, 48, 80, 15)
    assert preds[1].shape == (2, 1, 24, 40, 15)
    assert preds[2].shape == (2, 1, 12, 20, 15)
