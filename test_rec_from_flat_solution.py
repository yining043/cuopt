"""Quick test for rec_from_flat_solution: round-trip and explicit rec values."""
import torch
from CVRPEnv import rec_from_flat_solution


def rec_to_flat(rec: torch.Tensor, start: int = 0, max_steps: int = 100) -> list:
    out = [start]
    cur = start
    for _ in range(max_steps):
        nxt = rec[cur].item()
        out.append(nxt)
        if nxt == start and len(out) > 1:
            break
        cur = nxt
    return out


def main():
    # Single route: 0 -> 1 -> 2 -> 0
    flat1 = [0, 1, 2, 0]
    num_nodes = 3
    solutions = torch.tensor([flat1], dtype=torch.long)
    rec = rec_from_flat_solution(solutions, num_nodes, lengths=torch.tensor([4]))
    assert rec[0, 0].item() == 1 and rec[0, 1].item() == 2 and rec[0, 2].item() == 0
    assert rec_to_flat(rec[0]) == flat1
    print("single route:", flat1, "-> rec", rec[0].tolist(), "-> back", rec_to_flat(rec[0]))

    # Batch with padding
    flat_a, flat_b = [0, 1, 0], [0, 2, 1, 0]
    solutions = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(flat_a), torch.tensor(flat_b)], batch_first=True, padding_value=0
    )
    lengths = torch.tensor([3, 4])
    rec = rec_from_flat_solution(solutions, num_nodes, lengths=lengths)
    assert rec_to_flat(rec[0]) == flat_a and rec_to_flat(rec[1]) == flat_b
    print("batch with padding: ok")

    # lengths must be used so padding 0 does not overwrite rec[0]
    rec_len = rec_from_flat_solution(
        torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.long), num_nodes, lengths=torch.tensor([3])
    )
    assert rec_to_flat(rec_len[0]) == flat_a
    print("padding ignored when lengths=3: ok")

    print("all checks passed.")


if __name__ == "__main__":
    main()
