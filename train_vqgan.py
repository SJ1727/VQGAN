import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import lpips
from vqgan import VQGAN
from discriminator import Discriminator
import warnings
from image_dataset import ImageDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="VQGAN")
parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
parser.add_argument("--encoder-channels", type=int, nargs="+", default=[128, 128, 256, 256, 512], help="Number of channel for each layer in encoder")
parser.add_argument("--codebook-size", type=int, default=512, help="Number of codebook vectors")
parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss scalar")
parser.add_argument("--discriminator-layers", type=int, nargs="+", default=[64, 128], help="Layer sizes in discriminator")
parser.add_argument("--image-channels", type=int, default=3, help="Number of channels of images") # Might break if not 3 cuz lpips
parser.add_argument("--image-size", type=int, default=128, help="Image width and height")
parser.add_argument("--device", type=str, default="cpu", help="Training device")
parser.add_argument("--batch-size", type=int, default=16, help="Input batch size for training")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--learning-rate", type=float, default=2.25e-05, help="Learning rate")
parser.add_argument("--save-path", type=str, default="./checkpoints", help="Path to directory to save checkpoints")
parser.add_argument("--discriminator-threshold", type=int, default=1000, help="The number of training steps before discriminator starts")
parser.add_argument("--load-checkpoint-path", type=str, default=None, help="Path to load training checkpoint (will train new vqgan if path no specified)")
parser.add_argument("--images-directory", type=str, default=None, help="Path to image directory")
parser.add_argument("--tensorboard-session-path", type=str, default=None, help="Path to tensorboard session")

args = parser.parse_args()

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
model = VQGAN(args)
training_data = ImageDataset(args.images_directory, args.image_size, transform=transform)
training_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

step = 0

# Initilising model, discriminator, and perceptual loss
model = model.to(args.device)
discriminator = Discriminator(args).to(args.device)
perceptual_loss_fn = lpips.LPIPS(net="vgg", verbose=False).to(args.device)

# Initilising optimizers
optim_vqgan = optim.Adam(model.parameters(), lr=args.learning_rate)
optim_disc = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

# Resumes training
if args.load_checkpoint_path is not None:
    state = torch.load(args.load_checkpoint_path)

    step = state["step"]
    model.load_state_dict(state["model"])
    discriminator.load_state_dict(state["discriminator"])
    optim_vqgan.load_state_dict(state["model_optim"])
    optim_disc.load_state_dict(state["discriminator_optim"])
    args = state["args"]

writer = SummaryWriter(args.tensorboard_session_path)

print("---Training model---")
if args.load_checkpoint_path is not None:
    print(f"Resuming Training: {args.load_checkpoint_path}")
print(f"Tensorboard session: {writer.log_dir}")
print(f"Training Checkpoint: {args.save_path + rf'/training_checkpoint.pt'}")
print(f"Model Checkpoint: {args.save_path + rf'/model_checkpoint.pt'}")

averaging_steps = 0
total_vqgan_loss = 0
total_vq_loss = 0
total_disc_loss = 0

for epoch in range(args.epochs):
    for i, images in enumerate(training_dataloader):
        step += 1
        images = images.to(args.device)

        reconstruction, _, quant_loss = model(images)

        disc_real_prediction = discriminator(images)
        disc_fake_prediction = discriminator(reconstruction)

        reconstruction_loss = torch.dist(images, reconstruction)
        perceptual_loss = perceptual_loss_fn(images, reconstruction)
        perceptual_reconstruction_loss = torch.mean(reconstruction_loss + perceptual_loss)
        gan_loss = -torch.mean(disc_fake_prediction)

        discriminator_factor = 1 if step > args.discriminator_threshold else 0
        vq_loss = perceptual_reconstruction_loss + quant_loss + discriminator_factor * model.calculate_lambda(perceptual_reconstruction_loss, gan_loss) * gan_loss

        disc_loss_real = torch.mean(F.relu(1 - disc_real_prediction))
        disc_loss_fake = torch.mean(F.relu(1 + disc_fake_prediction))
        disc_loss = discriminator_factor * 0.5 * (disc_loss_real + disc_loss_fake)
        
        optim_vqgan.zero_grad()
        vq_loss.backward(retain_graph=True)

        optim_disc.zero_grad()
        disc_loss.backward(retain_graph=True)
        
        optim_vqgan.step()
        optim_disc.step()

        averaging_steps += 1
        total_vqgan_loss += vq_loss
        total_vq_loss += perceptual_reconstruction_loss + quant_loss
        total_disc_loss += disc_loss

        # Logging data after every 10 batches
        if step % 10 == 0:
            state = {
                "step" : step,
                "model" : model.state_dict(),
                "discriminator" : discriminator.state_dict(),
                "model_optim" : optim_vqgan.state_dict(),
                "discriminator_optim" : optim_disc.state_dict(),
                "args" : args,
            }

            # Saving model and training
            torch.save(state, args.save_path + rf"/training_checkpoint.pt")
            torch.save(state["model"], args.save_path + rf"/model_checkpoint.pt")

            if step % 50 == 0:
                # Writing images to Tensorboard
                writer.add_image("Original Images", images[0], step)
                writer.add_image("Reconstructed Images", reconstruction[0], step)

            # Writing losses to Tensorboard
            writer.add_scalar("VQGAN Loss", total_vqgan_loss / averaging_steps, step)
            writer.add_scalar("Reconstruction Loss", total_vq_loss / averaging_steps, step)
            writer.add_scalar("Discriminator Loss", total_disc_loss / averaging_steps, step)
            
            print(f"Step {step} | VQGAN Loss | Reconstruction Loss {total_vq_loss / averaging_steps} | Discriminator Loss {total_disc_loss/averaging_steps}")

            averaging_steps = 0
            total_vqgan_loss = 0
            total_vq_loss = 0
            total_disc_loss = 0

writer.close()