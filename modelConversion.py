import torch
import torch.nn as nn
# from LeNet import ModernLeNet5 as LeNet
from LeNetSmall import LeNet 
from OriginalLeNet import OGLeNet5 as OriginalLeNet
from PIL import Image
import coremltools as ct
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# import cv2

# LEARN
def preprocess_image(image_path):
    # Open and convert image to grayscale
    image = Image.open(image_path).convert('L')
    plt.imshow(image)
    plt.show()
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    
    # Use the custom transform function
    # image_tensor = transform(image_array)
    
    # Invert the tensor here (after conversion to tensor)
    image_tensor = 1 - image_tensor
    print(image_tensor, torch.min(image_tensor), torch.max(image_tensor))
    
    # Add batch dimension if needed (check shape first)
    if len(image_tensor.shape) == 3:  # If shape is (1, 32, 32)
        image_tensor = image_tensor.unsqueeze(0)  # Make it (1, 1, 32, 32)
    print(image_tensor.shape)
    return image_tensor

def load_and_run_model(model_path, input):
    try:
        # Load the model
        model = OriginalLeNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        
        # Convert input to tensor if it's not already
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input)
        
        # Ensure input is float
        input = input.float()
        
        # Run inference
        with torch.no_grad():
            output = model(input)
        print(output)         
        return torch.argmax(output, dim=1).item()
    
    except Exception as e:
        print(f"Error loading or running model: {str(e)}")
        return None

def convert_to_coreml(model_path):
    model = OriginalLeNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    example_input = torch.randn(1, 1, 32, 32)
    traced_model = torch.jit.trace(model, example_input)
    
    # Define input and output descriptions
    input_description = "Grayscale image input of size 32x32"
    output_description = "Probability distribution over 10 digit classes"
    example_output = torch.randn(1, 10)
    # Convert to CoreML with metadata
    coreML_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input_image", 
                            shape=example_input.shape)],
        outputs=[ct.TensorType(name="output_probabilities")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS13
    )
    # Save the model
    coreML_model.save("LeNet.mlpackage")
    
    # Print the spec to verify
    print("\nModel Spec:")
    print(coreML_model.get_spec())


if __name__ == "__main__":
    # Example usage
    model_path = "./model.pt"
    
    # Process the image
    # input_tensor = preprocess_image("./4.png")
    # Load and run the model
    # output = load_and_run_model(model_path, input_tensor)
    convert_to_coreml(model_path)
    
    # if output is not None:
    #     print("Model output:", output)
