import click
import torch


@click.command()
@click.option('--model', '-m', required=True, help='The model filename (*.pt)')
@click.option('--output', '-o', required=True, help='The output filename (*.pt)')
def strip_model(model: str, output: str):
    """Remove the optim data of PyTorch models."""
    model = torch.load(model, map_location='cpu')
    model['optim'] = None
    torch.save(model, output)


if __name__ == "__main__":
    strip_model()
