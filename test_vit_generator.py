import torch
from model import ViTGenerator

def test_vit_generator():
    print("Testing ViTGenerator...")
    try:
        generator = ViTGenerator(model_name='vit_tiny_patch16_224', pretrained=False)
        input_tensor = torch.randn(1, 3, 224, 224)
        output = generator(input_tensor)
        print(f"Output shape: {output.shape}")
        
        expected_shape = (1, 3, 224, 224)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("ViTGenerator test passed!")
    except Exception as e:
        print(f"ViTGenerator test failed: {e}")
        raise

if __name__ == "__main__":
    test_vit_generator()
