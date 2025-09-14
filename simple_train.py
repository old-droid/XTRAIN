"""
CPUWARP-ML Simple Training Script
==================================
The easiest way to train AI models!
"""

import sys
import time
from run_model import ModelRunner

def train_simple_llm():
    """Train a simple language model"""
    print("🤖 Training a Language Model (Mini ChatGPT)...")
    print("-" * 50)
    
    runner = ModelRunner('llm')
    runner.load_dataset('wikitext')
    
    print("📚 Training on text data...")
    runner.train(epochs=1, batch_size=2)
    
    print("✅ Language model trained successfully!")
    print("💾 Model saved to: model_llm_*.npz")
    return runner

def train_simple_cnn():
    """Train a simple image classifier"""
    print("🖼️ Training an Image Recognition Model...")
    print("-" * 50)
    
    runner = ModelRunner('cnn')
    runner.load_dataset('cifar10')
    
    print("📸 Training on image data...")
    runner.train(epochs=1, batch_size=4)
    
    print("✅ Image model trained successfully!")
    print("💾 Model saved to: model_cnn_*.npz")
    return runner

def train_emotion_ai():
    """Train an emotion understanding AI"""
    print("😊 Training an Emotion AI...")
    print("-" * 50)
    
    runner = ModelRunner('multimodal')
    runner.load_dataset('vqa')
    
    print("🎭 Training on multimodal data...")
    runner.train(epochs=1, batch_size=2)
    
    print("✅ Emotion AI trained successfully!")
    print("💾 Model saved to: model_multimodal_*.npz")
    return runner

def interactive_menu():
    """Interactive menu for beginners"""
    print("""
╔═══════════════════════════════════════════════════════╗
║          CPUWARP-ML - Simple AI Training              ║
╚═══════════════════════════════════════════════════════╝

Choose what type of AI you want to train:

1️⃣  Language Model (ChatGPT-like)
    - Generates text
    - Answers questions
    - Writes stories

2️⃣  Image Recognition
    - Identifies objects
    - Classifies images
    - Computer vision

3️⃣  Emotion AI
    - Understands emotions
    - Multimodal (text + images)
    - Sentiment analysis

4️⃣  Quick Test (Train all 3 in sequence)

5️⃣  Exit

""")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        print("\n" + "="*60)
        train_simple_llm()
        print("="*60)
        print("\n🎉 Your Language Model is ready to use!")
        print("Try: python -c \"from train_llm import CPUWarpTransformer; model = CPUWarpTransformer(vocab_size=1000, d_model=128, num_heads=4, num_layers=2)\"")
        
    elif choice == '2':
        print("\n" + "="*60)
        train_simple_cnn()
        print("="*60)
        print("\n🎉 Your Image Recognition Model is ready!")
        print("It can now classify images into categories!")
        
    elif choice == '3':
        print("\n" + "="*60)
        train_emotion_ai()
        print("="*60)
        print("\n🎉 Your Emotion AI is ready!")
        print("It can understand emotions from text and images!")
        
    elif choice == '4':
        print("\n🚀 Training all 3 models (this will take a few minutes)...")
        print("="*60)
        
        start = time.time()
        
        print("\n[1/3] Language Model")
        train_simple_llm()
        
        print("\n[2/3] Image Recognition")
        train_simple_cnn()
        
        print("\n[3/3] Emotion AI")
        train_emotion_ai()
        
        total_time = time.time() - start
        print("="*60)
        print(f"\n✨ All models trained in {total_time:.1f} seconds!")
        print("🎉 You now have 3 different AI models ready to use!")
        
    elif choice == '5':
        print("\n👋 Goodbye! Happy AI training!")
        sys.exit(0)
        
    else:
        print("\n❌ Invalid choice. Please try again.")
        interactive_menu()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple AI Training with CPUWARP-ML')
    parser.add_argument('--type', choices=['llm', 'cnn', 'emotion', 'all'], 
                       help='Type of model to train')
    parser.add_argument('--interactive', action='store_true',
                       help='Launch interactive menu')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.type):
        interactive_menu()
    elif args.type == 'llm':
        train_simple_llm()
    elif args.type == 'cnn':
        train_simple_cnn()
    elif args.type == 'emotion':
        train_emotion_ai()
    elif args.type == 'all':
        print("Training all models...")
        train_simple_llm()
        train_simple_cnn()
        train_emotion_ai()
        print("\n✅ All models trained successfully!")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════╗
    ║     Welcome to CPUWARP-ML Training!       ║
    ║         Easy AI for Everyone               ║
    ╚════════════════════════════════════════════╝
    """)
    
    main()