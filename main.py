import os
import time
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import cv2
import requests
from io import BytesIO
from transformers import RobertaTokenizer, RobertaModel
from timm import create_model  
import shutil
import psutil

# Paths for datasets
ratings_file = 'ratings.dat'
movies_file = 'movies.dat'
image_folder = 'images'
video_folder = 'videos'  
metadata_file = 'metadata.csv'

# API Key for TMDb
api_key = ""

# Initialize Models and Timer
start_time = time.time()

# Define Dataset Class
class MultimodalDataset(Dataset):
    def __init__(self, ratings_file, movies_file, image_folder, video_folder=None, metadata_file=None, transform=None):
        self.ratings_data = pd.read_csv(ratings_file, sep="::", names=["userID", "movieID", "rating","timeStamp"], engine='python', encoding='latin1')
        self.movies_data = pd.read_csv(movies_file, sep="::", names=["movieID", "title", "genres"], engine='python', encoding='latin1')
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.metadata_data = None
        if metadata_file and os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
            self.metadata_data = pd.read_csv(metadata_file)
        else:
            print("Warning: Metadata file not found or empty. Proceeding without metadata.")
        self.transform = transform or transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        
        # Initialize models
        try:
            self.efficientnet = create_model('efficientnetv2_s', pretrained=True).eval()
        except RuntimeError as e:
            print("EfficientNetV2_s pretrained weights unavailable. Falling back to EfficientNetB0.")
            self.efficientnet = create_model('efficientnet_b0', pretrained=True).eval()

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.roberta_model = RobertaModel.from_pretrained('roberta-base').eval()
        self.video_transformer = create_model('vit_base_patch16_224', pretrained=True).eval()

    def text2feature(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def image2feature(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        with torch.no_grad():
            features = self.efficientnet.forward_features(image.unsqueeze(0))
        return features

    def vid2feature(self, video_path, frame_count=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success and len(frames) < frame_count:
            frames.append(frame)
            success, frame = cap.read()
        cap.release()
        
        features = []
        for frame in frames:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.transform:
                pil_image = self.transform(pil_image)
            with torch.no_grad():
                frame_features = self.video_transformer(pil_image.unsqueeze(0))
            features.append(frame_features)
        return torch.stack(features)

    def __len__(self):
        return len(self.ratings_data)

    def __getitem__(self, idx):
        movieID = self.ratings_data.iloc[idx]["movieID"]
        movie_info = self.movies_data[self.movies_data["movieID"] == movieID].iloc[0]
        text = f"{movie_info['title']} {movie_info['genres']}"
        text_features = self.text2feature(text)  

        image_path = os.path.join(self.image_folder, f"{movieID}.jpg")
        image_features = torch.zeros(1, 1)  
        if os.path.exists(image_path):
            image_features = self.image2feature(image_path)  

        video_features = torch.zeros(1, 1)  
        if self.video_folder:
            video_path = os.path.join(self.video_folder, f"{movieID}.mp4")
            if os.path.exists(video_path):
                video_features = self.vid2feature(video_path).mean(dim=0).unsqueeze(0)  

        metadata_features = torch.zeros(1, 1)  
        if self.metadata_data is not None:
            metadata_features = torch.tensor(self.metadata_data.iloc[idx].values, dtype=torch.float32).unsqueeze(0)

       
        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        if image_features.dim() == 1:
            image_features = image_features.unsqueeze(0)
        if video_features.dim() == 1:
            video_features = video_features.unsqueeze(0)
        if metadata_features.dim() == 1:
            metadata_features = metadata_features.unsqueeze(0)

        concatenated_features = torch.cat((text_features, image_features, video_features, metadata_features), dim=1)
        return concatenated_features


dataset = MultimodalDataset(ratings_file, movies_file, image_folder, video_folder, metadata_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


features_start = time.time()
for i, data in enumerate(dataloader):
    if i > 2:  
        break
features_end = time.time()

def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Trainable Parameters: {param.numel()}")

print("Parameters trained at each layer of the model:")
print_model_parameters(dataset.efficientnet)

# Get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 ** 2  

initial_memory = get_memory_usage()
print(f"Initial Memory Usage: {initial_memory:.2f} MB")

# Calculate memory used by the model
total_memory_used = get_memory_usage() - initial_memory
print(f"Total Memory Used by Model: {total_memory_used:.2f} MB")

# Time Summary Table
model_initialization_time = features_start - start_time
feature_extraction_time = features_end - features_start
total_time = features_end - start_time

summary_table = pd.DataFrame({
    "Stage": ["Model Initialization", "Feature Extraction", "Total"],
    "Time (s)": [model_initialization_time, feature_extraction_time, total_time]
})

print(summary_table)
