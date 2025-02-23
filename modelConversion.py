import torch
import torch.nn as nn
from OriginalLeNet import OGLeNet5 as LeNet
from PIL import Image
def load_and_run_model(model_path, example_input):
    try:
        # Load the model
        model = LeNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode
        
        # Convert input to tensor if it's not already
        if not isinstance(example_input, torch.Tensor):
            example_input = torch.tensor(example_input)
        
        # Ensure input is float and has batch dimension
        example_input = example_input.float()
        if len(example_input.shape) == 1:
            example_input = example_input.unsqueeze(0)
            
        # Run inference
        with torch.no_grad():
            output = model(example_input)
            
        return torch.argmax(output, dim=1).item()
    
    except Exception as e:
        print(f"Error loading or running model: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    model_path = "./model.pt"
    image = Image.open("test_image.png")
    # Create example input data - LeNet expects 32x32 images with 1 channel
    example_data = torch.randn(1, 1, 32, 32)  # (batch_size, channels, height, width)
    
    # Load and run the model
    output = load_and_run_model(model_path, example_data)
    
    if output is not None:
        print("Model output:", output)
