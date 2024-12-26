import torch
from torchvision import transforms, models
from PIL import Image
import json
import argparse

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights='VGG16_Weights.DEFAULT')
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(weights='VGG13_Weights.DEFAULT')
    else:
        raise ValueError(f'Unsupported architecture: {checkpoint["arch"]}')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    return image

def predict(image_path, model, top_k, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    image = process_image(image_path).to(device)
    model.to(device)

    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(top_k, dim=1)

    top_class_names = [cat_to_name[str(cls)] for cls in top_class.cpu().numpy()[0]]
    return top_p.cpu().numpy()[0], top_class_names

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input, model, args.top_k, args.gpu)

    print("Probabilities:", probs)
    print("Classes:", classes)

if __name__ == '__main__':
    main()
