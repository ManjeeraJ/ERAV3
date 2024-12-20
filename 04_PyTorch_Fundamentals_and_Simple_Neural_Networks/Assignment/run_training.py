import os
from train import TrainingManager
import json
from tqdm import tqdm

def main():
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    training_manager = TrainingManager()
    
    num_epochs = 2
    print("\nStarting training...")
    for epoch in tqdm(range(num_epochs), desc='Overall Progress', position=0):
        training_manager.train_epoch()
    
    print("\nTraining completed! Evaluating model...")
    # Save final results
    test_loss, accuracy = training_manager.evaluate()
    samples = training_manager.get_random_samples()
    
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'samples': samples
    }
    
    with open('static/results.json', 'w') as f:
        json.dump(results, f)
    
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main() 