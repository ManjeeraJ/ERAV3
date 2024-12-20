import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTNet
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

class TrainingManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MNISTNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.train_losses = []
        self.current_epoch = 0
        self.current_batch = 0
        
        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True, transform=transform),
            batch_size=64, shuffle=True)
        
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transform),
            batch_size=1000)

    def train_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), 
                   total=len(self.train_loader),
                   desc=f'Epoch {self.current_epoch}',
                   leave=True)
        
        for batch_idx, (data, target) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            self.train_losses.append(loss.item())
            self.current_batch = batch_idx
            
            # Update progress bar with loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if batch_idx % 10 == 0:
                self.save_training_state()
                
        self.current_epoch += 1
        return self.train_losses[-1]

    def save_training_state(self):
        state = {
            'epoch': self.current_epoch,
            'batch': self.current_batch,
            'losses': self.train_losses
        }
        with open('static/training_state.json', 'w') as f:
            json.dump(state, f)

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return test_loss, accuracy

    def get_random_samples(self):
        self.model.eval()
        dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
        indices = torch.randperm(len(dataset))[:10]
        samples = []
        
        for idx in indices:
            image, true_label = dataset[idx]
            with torch.no_grad():
                output = self.model(image.unsqueeze(0).to(self.device))
                pred_label = output.argmax(dim=1).item()
                
            samples.append({
                'image': image.numpy()[0].tolist(),
                'true_label': int(true_label),
                'pred_label': int(pred_label)
            })
            
        return samples 