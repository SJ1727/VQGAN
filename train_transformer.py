import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from transformer import VQGANTransformer
import warnings
from image_dataset import ImageDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Transformer")
parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
parser.add_argument("--codebook-size", type=int, default=1024, help="Number of codebook vectors")
parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 128, 256, 256, 512], help="Number of channel for each layer in encoder")
parser.add_argument("--discriminator-layers", type=int, nargs="+", default=[64, 128], help="Layer sizes in discriminator (default is a 16x16)")
parser.add_argument("--image-channels", type=int, default=3, help="Number of channels of images")
parser.add_argument("--image-size", type=int, default=64, help="Image width and height")
parser.add_argument("--device", type=str, default="cpu", help="Training device")
parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss scalar")
parser.add_argument("--batch-size", type=int, default=16, help="Input batch size for training")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--learning-rate", type=float, default=2.25e-05, help="Learning rate")
parser.add_argument("--save-path", type=str, default="./checkpoints", help="Path to directory to save checkpoints")
parser.add_argument("--vqgan-path", type=str, default=None, help="Path to vqgan")
parser.add_argument("--images-directory", type=str, default=None, help="Path to image directory")
parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads in transformer")
parser.add_argument("--transformer-layers", type=int, default=2, help="Number of layers in transformer")
parser.add_argument("--feed-forward-dim", type=int, default=256, help="Dimension of feed forward in transformer")
parser.add_argument("--feed-forward-dropout", type=float, default=0.1, help="Dropout rate in feed forward layer")
parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout rate in attention layer")
parser.add_argument("--out-dropout", type=float, default=0.1, help="Dropout rate for the output layer")
parser.add_argument("--sos-token", type=float, default=1.0, help="Value to be used for sos token")
parser.add_argument("--tensorboard-session-path", type=str, default=None, help="Path to tensorboard session")
parser.add_argument("--top-k", type=int, default=50, help="Top k values to be selected when generating a new image")

args = parser.parse_args()

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
training_data = ImageDataset(args.images_directory, args.image_size, transform=transform)
training_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

model = VQGANTransformer(args)
criterion = nn.CrossEntropyLoss()
transformer_optim = optim.Adam(model.parameters(), lr=args.learning_rate)

model = model.to(args.device)

step = 0

writer = SummaryWriter(args.tensorboard_session_path)

for i, image in enumerate(training_dataloader):
    step += 1
    
    image = image.to(args.device)

    transformer_optim.zero_grad()
    
    predictions, actual = model(image)
    predictions = predictions[:, :-1, :]
    
    loss = criterion(predictions, actual)
    loss.backward()
    
    transformer_optim.step()
    
    if i % 10 == 0:
        # Saving model
        torch.save(model.state_dict(), args.save_path + rf"/model_transformer_checkpoint.pt")
        
        if i % 50 == 0:
            # Generating new image
            image = model.generate(16)
            
            # Writing image to Tensorboard
            writer.add_image("Generated Images", image[0], step)

        # Writing loss to Tensorboard
        writer.add_scalar("Transformer Loss", loss, step)
        
        print(f"Step {step} | Loss {loss}")

writer.close()

