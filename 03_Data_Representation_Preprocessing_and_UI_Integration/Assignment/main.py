from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.corpus import wordnet
import random

# Download required NLTK data
nltk.download('wordnet')

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

class TextDataset(Dataset):
    def __init__(self, text, max_length=10):
        self.lines = [text]
        self.tokenizer = get_tokenizer('basic_english')
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx].strip()
        tokens = self.tokenizer(text)
        token_ids = torch.tensor([ord(token[0]) for token in tokens], dtype=torch.long)
        
        if len(token_ids) < self.max_length:
            token_ids = torch.cat([token_ids, torch.zeros(self.max_length - len(token_ids), dtype=torch.long)])
        else:
            token_ids = token_ids[:self.max_length]
        
        return token_ids, 0

def preprocess_text(text):
    dataset = TextDataset(text)
    token_ids, _ = dataset[0]
    return f"Preprocessed: {token_ids.tolist()}"

def augment_text(text):
    words = text.split()
    n = max(1, len(words) // 3)  # Replace about 1/3 of words
    for _ in range(n):
        word_to_replace = random.choice(words)
        synonyms = wordnet.synsets(word_to_replace)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            words = [synonym if word == word_to_replace else word for word in words]
    return ' '.join(words)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile):
    content = await file.read()
    text = content.decode()
    # Split into lines and filter out empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Select a random line
    random_line = random.choice(lines)
    return {"text": random_line}

@app.post("/preprocess")
async def preprocess(text: str = Form(...)):
    processed_text = preprocess_text(text)
    return {"processed_text": processed_text}

@app.post("/augment")
async def augment(text: str = Form(...)):
    augmented_text = augment_text(text)
    return {"augmented_text": augmented_text} 