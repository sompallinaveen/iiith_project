import torch


def collate_fn(batch):

    audios = []
    labels = []

    max_len = max(
        len(audio)
        for audio, _ in batch
    )

    for audio, label in batch:

        padded = torch.zeros(
            max_len
        )

        padded[:len(audio)] = audio

        audios.append(
            padded
        )

        labels.append(
            label
        )

    return (
        torch.stack(audios),
        torch.tensor(labels)
    )