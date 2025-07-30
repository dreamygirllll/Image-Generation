import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

def create_diffusion_pipeline(model_name="runwayml/stable-diffusion-v1-5"):
    """
    Creates and returns a Stable Diffusion pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype
    ).to(device)

    return pipe, device


def generate_images(pipe, device, prompts):
    """
    Generates images from a list of prompts using the given pipeline.
    """
    images = []
    for prompt in prompts:
        with torch.autocast(device):
            image = pipe(prompt).images[0]
            images.append((prompt, image))
    return images


def plot_images(images):
    """
    Plots images using matplotlib with their respective prompts as titles.
    """
    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))

    if len(images) == 1:
        axes = [axes]  # make it iterable if only one image

    for ax, (prompt, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(prompt, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ========== MAIN ==========
if __name__ == "__main__":
    # Ask user for prompts
    print("Enter your prompts (comma-separated):")
    user_input = input(">> ")
    prompts = [p.strip() for p in user_input.split(",") if p.strip()]

    if not prompts:
        print("No valid prompts provided.")
        exit()

    # Create pipeline
    pipeline, device = create_diffusion_pipeline()

    # Generate and plot images
    generated_images = generate_images(pipeline, device, prompts)
    plot_images(generated_images)
