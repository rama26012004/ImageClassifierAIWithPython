import torch
from torchvision import transforms
from PIL import Image
import json

model.load_state_dict(torch.load('checkpoint.pth'))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image).unsqueeze(0)
    return image

def predict(image_path, model, top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = process_image(image_path).to(device)
    model.to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)
        top_class_names = [cat_to_name[str(cls)] for cls in top_class.cpu().numpy()[0]]
        return top_p.cpu().numpy()[0], top_class_names

image_path = 'flowers/test/1/image_06743.jpg'
probs, classes = predict(image_path, model)
print("Probabilities:", probs)
print("Classes:", classes)
