import torch
import json
from torch import optim
from torchvision import transforms, models
from PIL import Image

def process_image(image):
    im = Image.open(image) 
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return image_transforms(im)

def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def check_device(gpu):
    
    if not gpu:
        return torch.device("cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA unavailable. Falling back to use CPU.")
        
    return device

def load_checkpoint(filename, arch, gpu):
    loaddata = torch.load(filename)
    
    model_fn = getattr(models, loaddata['arch'])
    
    model = model_fn()
    model.classifier = loaddata['classifier']
    device = check_device(gpu)
    model.to(device)
    model.class_to_idx = loaddata['class_to_idx']
    model.load_state_dict(loaddata['model_state_dict'])
    

    optimizer = optim.Adam(model.classifier.parameters(), lr=loaddata['learning_rate'])
    optimizer.load_state_dict(loaddata['optimizer_state'])
    
    return model, optimizer