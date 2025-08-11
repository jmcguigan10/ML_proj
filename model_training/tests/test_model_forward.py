def test_forward_smoke():
    m = create_model(name="your_model", in_dim=8, hidden_dims=[16], out_dim=3)
    x = torch.randn(4, 8)
    y = m(x)
    assert y.shape == (4, 3) and torch.isfinite(y).all()
