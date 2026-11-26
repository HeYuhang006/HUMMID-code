"""Run the CNN ablation with its own local checkpoint pair."""

from .Inference import main


if __name__ == "__main__":
    main(cnn=True)
