from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import json

from ..module_quantizer import ModuleQuantizer


##############################
# test for module quantizer for resnet18
# outputs correct class for specific set of quantization parameters (probably needs to be trained)
##############################

def test_resnet18__module_quantizer():
    # Load and preprocess a sample image to test the model
    image_path = 'dog.png'  # Replace with the path to your image
    image = Image.open(image_path).convert('RGB')

    # Define the transformation to preprocess the image for ResNet
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the preprocessing transformation to the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # load resnet model from torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    quantized_model = ModuleQuantizer(model, to_train=(True, False, True))

    with torch.no_grad():
        output = model(input_batch)
        output_with_q = quantized_model(input_batch)

    # Get the predicted class index
    _, predicted_class = output.max(1)
    _, predicted_class_q = output_with_q.max(1)

    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)

    # Print the predicted class label
    print(f"Predicted class from unqantized model: -- {labels[predicted_class.item()]} -- with probability: {torch.nn.functional.softmax(output, dim=1)[0][predicted_class.item()].item()}")
    print(f"Predicted class from quantized model: -- {labels[predicted_class_q.item()]} -- with probability: {torch.nn.functional.softmax(output_with_q, dim=1)[0][predicted_class_q.item()].item()}")


if __name__ == '__main__':
    test_resnet18__module_quantizer()