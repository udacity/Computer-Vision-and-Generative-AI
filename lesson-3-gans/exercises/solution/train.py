import random
import argparse
from pathlib import Path

from torch import nn, optim
import torch
import numpy as np
import tqdm

from data import get_dataloader
from viz import training_tracking
from diff_augment import DiffAugment
from generator import Generator
from discriminator import Discriminator


manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results


def get_optimizer_and_loss(D, G, lr, beta1):
    """
    Returns the optimizer to be used for training.

    :param D: discriminator network
    :param G: generator network
    :param lr: learning rate
    :param beta1: beta1 parameter for Adam optimizer
    """
    # Setup optimization
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    return D_optimizer, G_optimizer, criterion


def weights_init(m):
    """
    Initialize Conv layers and BatchNorm from a normal distribution with mean 0 and std 0.02.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_positive_labels(size, device, smoothing=True, random_flip=0.05):
    if smoothing:
        # Random positive numbers between 0.8 and 1.2 (label smoothing)
        labels = 0.8 + 0.4 * torch.rand(size, device=device)
    else:
        labels = torch.full((size,), 1.0, device=device)

    if random_flip > 0:
        # Let's flip some of the labels to make it slightly harder for the discriminator
        num_to_flip = int(random_flip * labels.size(0))

        # Get random indices and set the first "num_to_flip" of them to 0
        indices = torch.randperm(labels.size(0))[:num_to_flip]
        labels[indices] = 0

    return labels


def get_negative_labels(size, device):
    return torch.full((size,), 0.0, device=device)


def train(
    data_root_path: Path,
    image_size: int,
    num_channels: int,
    latent_dimension: int,
    batch_size: int,
    lr: float,
    beta1: float,
    D_feat_map_size: int,
    G_feat_map_size: int,
    n_epochs: int,
    policy: str="color,translation,cutout",
    smoothing: bool=True,
    random_flip: float=0.05,
    gradient_clipping_value: float=0,
    jupyter: bool=False,
    output_dir: Path="training_results",
    save_iter: int=10,
    D_dropout: float=0.2
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Let's generate a latent vector of fixed noise to visualize the progression
    # of the Generator
    fixed_noise = torch.randn(16, latent_dimension, 1, 1, device=device)

    # Get dataloader
    dataloader = get_dataloader(data_root_path, image_size, batch_size)

    # Setup model and optimizer
    G = Generator(image_size, latent_dimension, G_feat_map_size, num_channels)
    D = Discriminator(image_size, D_feat_map_size, num_channels, dropout=D_dropout)

    D_optimizer, G_optimizer, criterion = get_optimizer_and_loss(D, G, lr, beta1)

    D.to(device)
    D.apply(weights_init)
    G.to(device)
    G.apply(weights_init)

    # Lists to keep track of progress
    G_losses = []
    D_losses = []
    D_acc = []

    print("Training is starting...")

    # For each epoch
    for epoch in range(n_epochs):
        # Accumulate the losses for each batch in these lists
        batch_G_losses = []
        batch_D_losses = []
        # Accumulate the accuracy for each batch in this list
        batch_D_accuracy = []

        # We will train the Discriminator on a batch that is made by
        # half real data and half fake data. Then we will train the
        # Generator on a batch of only fake data.
        # Remember that we have configured the dataloader to return
        # batches of half batch_size, so the total batch size that the
        # Discriminator sees is batch_size (half real and half fake data)
        # just like the Generator

        for real_data_batch in tqdm.tqdm(dataloader):
            
            D.zero_grad()
            # Format batch
            real_data_batch = real_data_batch[0].to(device)
            b_size = real_data_batch.size(0)
            
            # Random positive numbers between 0.8 and 1.2 (label smoothing)
            labels = 0.8 + 0.4 * torch.rand(b_size, device=device)
            
            # Let's flip 10% of the labels to make it slightly harder for the discriminator
            num_to_flip = int(0.05 * labels.size(0))

            # Get random indices and set the first "num_to_flip" of them to 0
            indices = torch.randperm(labels.size(0))[:num_to_flip]
            labels[indices] = 0
            
            # Forward pass real batch through D
            output = D(DiffAugment(real_data_batch, policy=policy)).view(-1)
            
            # Accuracy for positive
            acc_pos = (output > 0.5).sum() / output.size(0)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, labels)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent_dimension, 1, 1, device=device)
            # Generate fake image batch with G
            fake = G(noise)
            labels.fill_(0.0)

            # Classify all fake batch with D
            output = D(DiffAugment(fake, policy=policy).detach()).view(-1)
            
            # Accuracy for negative
            acc_neg = (output < 0.5).sum() / output.size(0)
            
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, labels)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()

            # Compute error of D as sum over the fake and the real batches
            D_loss = errD_real + errD_fake
            # Gradient clipping
            #nn.utils.clip_grad_value_(D.parameters(), clip_value=1.0)
            # Update D
            D_optimizer.step()

            # Store current loss
            batch_D_losses.append(D_loss.item())
            batch_D_accuracy.append(((acc_pos + acc_neg) / 2).item())

            # Generator training
            G.zero_grad()

            # Trick suggested by Goodfellow et al. in the original GAN paper:
            # we want to train G to maximize log(D(G(z))) instead of minimizing
            # log(1 - D(G(z))) because the latter can saturate. This is obtained by
            # tricking the loss function to use the second part of the BCELoss
            # (i.e. log(x)) instead of the first one (i.e. log(1-x))
            labels.fill_(1.0)

            # D just changed in the previous step, so we need to do another forward pass
            # through D to get the new predictions on the fake data we already generated
            # as well as the new one
            D_pred = D(
                DiffAugment(fake, policy=policy)
            ).view(-1)
            # Calculate G's loss based on this output
            G_loss = criterion(D_pred, labels)
            # Calculate gradients for G
            G_loss.backward()

            if gradient_clipping_value > 0:
                # Gradient clipping
                nn.utils.clip_grad_value_(
                    G.parameters(), clip_value=gradient_clipping_value
                )

            # Update G
            G_optimizer.step()

            # Save Losses for plotting later
            batch_G_losses.append(G_loss.item())

        G_losses.append(np.mean(batch_G_losses))
        D_losses.append(np.mean(batch_D_losses))
        D_acc.append(np.mean(batch_D_accuracy))

        if not jupyter:
            # Print losses and accuracy
            print(f"Epoch {epoch+1} / {n_epochs}")
            print(
                f"G loss: {G_losses[-1]:8.2f} | D loss: {D_losses[-1]:8.2f} | D accuracy: {D_acc[-1]:4.3f}"
            )

        if epoch % save_iter == 0:
            with torch.no_grad():
                fake_viz_data = G(fixed_noise).detach().cpu()

            if jupyter:
                from IPython.display import clear_output
                import matplotlib.pyplot as plt

                clear_output(wait=True)

                fig = training_tracking(D_losses, G_losses, D_acc, fake_viz_data)

                plt.show()

            else:
                fig = training_tracking(D_losses, G_losses, D_acc, fake_viz_data)

            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_dir / f"training_tracking_{epoch}.png", bbox_inches="tight"
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training script")

    parser.add_argument("--data-root-path", type=Path, required=True, help="Path to the data root")
    parser.add_argument("--image-size", type=int, required=True, help="Size of the image")
    parser.add_argument("--num-channels", type=int, required=True, help="Number of channels in the image")
    parser.add_argument("--latent-dimension", type=int, required=True, help="Latent dimension size")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--beta1", type=float, required=True, help="Beta1 value for the Adam optimizer")
    parser.add_argument("--D-feat-map-size", type=int, required=True, help="Feature map size for Discriminator")
    parser.add_argument("--G-feat-map-size", type=int, required=True, help="Feature map size for Generator")
    parser.add_argument("--n-epochs", type=int, required=True, help="Number of epochs for training")

    # Default parameters
    parser.add_argument("--D-dropout", type=float, required=False, help="Dropout for the Discriminator", default=0.2)
    parser.add_argument("--policy", type=str, default="color,translation,cutout", help="Policy types")
    parser.add_argument("--smoothing", action='store_true', help="Enable label smoothing")
    parser.add_argument("--random-flip", type=float, default=0.05, help="Random flip probability")
    parser.add_argument("--gradient-clipping-value", type=float, default=0, help="Gradient clipping value")
    parser.add_argument("--jupyter", action='store_true', help="Running in Jupyter environment")
    parser.add_argument("--output-dir", type=Path, default="training_results", help="Directory to save output results")
    parser.add_argument("--save-iter", type=int, default=10, help="Frequency of saving results")

    args = parser.parse_args()
    
    print(vars(args))

    train(**vars(args))
