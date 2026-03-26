"""
Evaluation utilities for FACH:
  - calc_hamming_dist : pairwise Hamming distance matrix
  - calc_map_k        : mean Average Precision @ K  (standard mAP)
  - calc_targeted_map : targeted mAP (t-mAP) when target labels differ from query
"""

import torch


def calc_hamming_dist(B1: torch.Tensor, B2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise Hamming distance between rows of B1 (Q×K) and B2 (R×K).
    Returns (Q, R) matrix.
    """
    K = B2.shape[1]
    if B1.dim() == 1:
        B1 = B1.unsqueeze(0)
    # Hamming distance via inner product on {-1,+1} codes
    return 0.5 * (K - B1.mm(B2.t()))


def calc_map_k(
    qB: torch.Tensor,
    rB: torch.Tensor,
    query_label: torch.Tensor,
    retrieval_label: torch.Tensor,
    k: int = None,
) -> torch.Tensor:
    """
    Mean Average Precision @ k  (Eq. 21).

    Args:
        qB, rB            : (Q, K) and (R, K) binary hash codes in {-1, +1}.
        query_label       : (Q, C) multi-hot label matrix.
        retrieval_label   : (R, C) multi-hot label matrix.
        k                 : number of top results to consider (None → all R).

    Returns:
        mAP scalar (Python float).
    """
    num_query = query_label.shape[0]
    if k is None:
        k = retrieval_label.shape[0]

    map_val = 0.0
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = gnd.sum()
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i], rB)
        _, ind = torch.sort(hamm.squeeze())
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1, dtype=torch.float, device=gnd.device)
        tindex = torch.nonzero(gnd, as_tuple=False)[:total].squeeze().float() + 1.0
        if tindex.dim() == 0:
            tindex = tindex.unsqueeze(0)
            count  = count[:1]
        map_val += (count / tindex).mean()

    return map_val / num_query


def calc_targeted_map(
    qB: torch.Tensor,
    rB: torch.Tensor,
    target_label: torch.Tensor,
    retrieval_label: torch.Tensor,
    k: int = None,
) -> torch.Tensor:
    """
    Targeted mAP: measure how well adversarial queries match *target* labels.

    Uses the same mAP formula but the relevance ground-truth is defined by
    target_label instead of the original query label.
    """
    return calc_map_k(qB, rB, target_label, retrieval_label, k)
