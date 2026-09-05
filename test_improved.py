"""
Test script for NovaCore Complete Neural Engine.
Tests internal neural network + python terminal + virtual simulation + verification.
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add NovaCore to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from novacore.inference.chat import ChatSession


def test_model(weights_dir, model_name):
    """Test a specific model with various prompts."""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        session = ChatSession(weights_dir)
        session.load()
        
        print(f"[OK] Model loaded successfully")
        print(f"  Documents: {session.metadata.get('docs', 'N/A')}")
        print(f"  Vocabulary: {len(session.predictor.vocab)} tokens")
        print(f"  Reservoir samples: {len(session.reservoir_samples)}")
        
        # Show model info
        model_info = session.get_model_info()
        print(f"  Model: {model_info.get('name', 'NovaCore')}")
        print(f"  Type: {model_info.get('type', 'Neural LLM')}")
        print(f"  Components: {', '.join(model_info.get('components', []))}")
        print(f"  Parameters: {model_info.get('parameters', 0):,}")
        
        # Show neural network visualization
        print(f"\n  Neural Network:")
        nn_viz = session.visualize_neural_network()
        for line in nn_viz.split('\n'):
            print(f"  {line}")
        
        # Test prompts
        test_prompts = [
            "hello",
            "what is machine learning?",
            "tell me a story",
            "calculate 2 plus 2",
            "explain python programming",
            "write a poem about nature",
        ]
        
        print(f"\nTesting {len(test_prompts)} prompts:")
        print("-" * 40)
        
        for prompt in test_prompts:
            print(f"\nYou> {prompt}")
            response = session.generate(prompt)
            print(f"NovaCore> {response}")
            
    except Exception as e:
        print(f"[ERR] Error testing {model_name}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("NovaCore Improved Generation Test")
    print("=" * 60)
    
    # Test both models
    models = [
        ("weights/TestV1", "TestV1 (dim=256, layers=2)"),
        ("weights/TestV2", "TestV2 (dim=512, layers=6)"),
    ]
    
    for weights_dir, model_name in models:
        if os.path.exists(weights_dir):
            test_model(weights_dir, model_name)
        else:
            print(f"✗ Model not found: {weights_dir}")
    
    print(f"\n{'='*60}")
    print("Test completed!")


if __name__ == "__main__":
    main()