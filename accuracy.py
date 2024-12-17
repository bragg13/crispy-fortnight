import torch
def token_lvl_accuracy(word2idx_tgt, gt, pred):
    """
    gt = ground truth sequence
    pred = predicted sequence
    """
    correct = 0
    
    # get start and end
    eos_idx = word2idx_tgt['<EOS>']
    sos_idx = word2idx_tgt['<SOS>']
    # print(eos_idx)
    # print(sos_idx)
    pred = pred[-1]


    gt = gt[-1]

    # index of <SOS> and <EOS> tokens of the predicted sequence
    pred_start = 0
    pred_end = len(pred) if (eos_idx not in pred) else (pred == eos_idx).nonzero(as_tuple=True)[0].item()

    # index of <SOS> and <EOS> tokens of the ground truth sequence
    gt_start = (gt == sos_idx).nonzero(as_tuple=True)[0].item()
    gt_end = (gt == eos_idx).nonzero(as_tuple=True)[0].item()

    # slicing
    gt = gt[gt_start+1 : gt_end]
    pred = pred[pred_start+1 : pred_end]

    longer = gt if len(gt) > len(pred) else pred
    shorter = pred if len(gt) > len(pred) else gt

    longest_len = len(longer)

    shorter = torch.nn.functional.pad(shorter, (0, longest_len - len(shorter)), "constant", 0)

    correct = sum(longer == shorter)
    # print(longer)
    # print(shorter)
    # print(correct)
    return int(correct) / len(shorter) # same length as longer


def sequence_level_accuracy(gt, pred, word2idx_tgt):

    # get start and end
    eos_idx = word2idx_tgt['<EOS>']
    sos_idx = word2idx_tgt['<SOS>']
    # print(eos_idx)
    # print(sos_idx)
    pred = pred[-1]
    gt = gt[-1]

    # index of <SOS> and <EOS> tokens of the predicted sequence
    pred_start = 0
    pred_end = len(pred) if (eos_idx not in pred) else (pred == eos_idx).nonzero(as_tuple=True)[0].item()

    # index of <SOS> and <EOS> tokens of the ground truth sequence
    gt_start = (gt == sos_idx).nonzero(as_tuple=True)[0].item()
    gt_end = (gt == eos_idx).nonzero(as_tuple=True)[0].item()

    # slicing
    gt = gt[gt_start+1 : gt_end]
    pred = pred[pred_start+1 : pred_end]

    if len(gt) != len(pred):
        return 0

    if sum(gt == pred) == len(gt):
        return 1

    return 0


def batched_token_lvl_accuracy(word2idx_tgt, gt_batch, pred_batch):
    """
    Calculate token-level accuracy for a batch of sequences.

    Args:
        word2idx_tgt: Dictionary mapping target vocabulary to indices.
        gt_batch: Ground truth sequences (Tensor of shape [batch_size, tgt_len]).
        pred_batch: Predicted sequences (Tensor of shape [batch_size, tgt_len]).

    Returns:
        Tensor of shape [batch_size] with token-level accuracy for each sample.
    """
    eos_idx = word2idx_tgt['<EOS>']
    sos_idx = word2idx_tgt['<SOS>']
    
    batch_size, _ = gt_batch.size(0)
    token_accuracies = []

    for i in range(batch_size):
        # Extract individual sequences
        gt = gt_batch[i]
        pred = pred_batch[i]

        # Get indices of start (<SOS>) and end (<EOS>) tokens
        print(pred)
        print(gt)
        pred_end = (pred == eos_idx).nonzero(as_tuple=True)
        pred_end = pred_end[0].item() if len(pred_end) > 0 else len(pred)
        gt_start = (gt == sos_idx).nonzero(as_tuple=True)[0].item()
        gt_end = (gt == eos_idx).nonzero(as_tuple=True)[0].item()


        # Slice to remove padding and SOS/EOS
        gt = gt[gt_start + 1: gt_end]
        pred = pred[1: pred_end]

        # Pad the shorter sequence
        longer_len = max(len(gt), len(pred))
        gt = torch.nn.functional.pad(gt, (0, longer_len - len(gt)), value=0)
        pred = torch.nn.functional.pad(pred, (0, longer_len - len(pred)), value=0)

        # Compute token-wise correctness
        correct = torch.sum(gt == pred).sum().item()
        token_accuracies.append(correct / longer_len)

    return torch.tensor(token_accuracies)

def batched_sequence_level_accuracy(gt_batch, pred_batch):
    """
    Calculate sequence-level accuracy for a batch of sequences.

    Args:
        gt_batch: Ground truth sequences (Tensor of shape [batch_size, tgt_len]).
        pred_batch: Predicted sequences (Tensor of shape [batch_size, tgt_len]).

    Returns:
        Tensor of shape [batch_size] with sequence-level accuracy for each sample (1 for exact match, 0 otherwise).
    """
    batch_size, _ = gt_batch.size()
    seq_accuracies = []

    for i in range(batch_size):
        # Extract individual sequences
        gt = gt_batch[i]
        pred = pred_batch[i]

        # Exact match
        if torch.equal(gt, pred):
            seq_accuracies.append(1)
        else:
            seq_accuracies.append(0)

    return torch.tensor(seq_accuracies)
