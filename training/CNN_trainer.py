"""Train the CNN ablation under the same data and stage interfaces as HUMMID."""

from .trainer import main


if __name__ == "__main__":
    main(cnn=True)
