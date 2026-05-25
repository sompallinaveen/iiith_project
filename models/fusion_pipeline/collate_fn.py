import torch
from torch.nn.utils.rnn import pad_sequence


def fusion_collate_fn(batch):

    audios = []
    input_ids = []
    attention_masks = []
    labels = []

    for audio, ids, mask, label in batch:

        audios.append(audio)

        input_ids.append(ids)

        attention_masks.append(mask)

        labels.append(label)

    audios = pad_sequence(
        audios,
        batch_first=True
    )

    input_ids = torch.stack(
        input_ids
    )

    attention_masks = torch.stack(
        attention_masks
    )

    labels = torch.stack(
        labels
    )

    return (
        audios,
        input_ids,
        attention_masks,
        labels
    )