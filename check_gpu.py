"""
Quick script to check GPU availability and configuration
"""
import torch

print("=" * 70)
print("🔍 GPU Configuration Check")
print("=" * 70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n📊 GPU {i}:")
        print(f"   Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print(f"   Multi-Processor Count: {props.multi_processor_count}")
    
    # Test tensor allocation
    print("\n🧪 Testing GPU tensor allocation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU tensor operations working!")
        print(f"   Test tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
else:
    print("\n⚠️  CUDA is not available!")
    print("   Reasons could be:")
    print("   1. PyTorch was installed without CUDA support")
    print("   2. No NVIDIA GPU detected")
    print("   3. CUDA drivers not installed")
    print("\n   To install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 70)
