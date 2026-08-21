import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, BertModel

from rlhf_scratch.data import PreferenceDataset, PreferencePair, make_collate_fn
from rlhf_scratch.models import RewardModel
from rlhf_scratch.training import RewardConfig, RewardTrainer, bradley_terry_loss


def _tiny_reward_model_and_tokenizer():
    """Tiny randomly-initialized BERT encoder standing in for bert-tiny, so
    tests run fast on CPU without downloading real weights.

    NOTE: uses bert-base-uncased's tokenizer, not prajjwal1/bert-tiny's — this
    transformers version cannot build a fast tokenizer from bert-tiny's legacy
    vocab.txt-only repo (raises "Couldn't instantiate the backend tokenizer").
    This is a real compatibility issue that will also block the actual
    Phase 8 training run and needs a fix (pin an older transformers, or use
    an alternative tiny BERT repo that ships tokenizer.json) before then.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    model = RewardModel.__new__(RewardModel)
    nn.Module.__init__(model)
    model.encoder = BertModel(config)
    model.value_head = nn.Linear(config.hidden_size, 1)
    return model, tokenizer


def test_bradley_terry_loss_prefers_correctly_ranked_pairs():
    chosen = torch.tensor([2.0, 0.5])
    rejected = torch.tensor([-1.0, 0.4])
    correctly_ranked_loss = bradley_terry_loss(chosen, rejected)

    reversed_loss = bradley_terry_loss(rejected, chosen)
    assert correctly_ranked_loss.item() < reversed_loss.item()
    assert correctly_ranked_loss.item() > 0  # -log(sigmoid(x)) is always positive


def test_bradley_terry_loss_symmetric_case_is_log2():
    chosen = torch.tensor([0.0])
    rejected = torch.tensor([0.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert torch.isclose(loss, torch.log(torch.tensor(2.0)), atol=1e-5)


def test_reward_model_forward_shape():
    model, tokenizer = _tiny_reward_model_and_tokenizer()
    enc = tokenizer(["hello there", "a longer sentence here"], padding=True, return_tensors="pt")
    rewards = model(enc["input_ids"], enc["attention_mask"])
    assert rewards.shape == (2,)


def test_reward_trainer_reduces_loss_and_evaluates_accuracy():
    model, tokenizer = _tiny_reward_model_and_tokenizer()
    collate_fn = make_collate_fn(tokenizer, max_length=32)

    pairs = [
        PreferencePair(prompt="Q: capital of France?\nA:", chosen=" Paris", rejected=" I refuse to answer"),
        PreferencePair(prompt="Q: 2+2?\nA:", chosen=" 4", rejected=" purple elephants"),
        PreferencePair(prompt="Q: sky color?\nA:", chosen=" blue", rejected=" asdkjaslkdj"),
    ]

    class _ListDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, idx):
            return pairs[idx]

    dataloader = DataLoader(_ListDataset(), batch_size=3, collate_fn=collate_fn)

    device = torch.device("cpu")
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer = RewardTrainer(model, optimizer, device, RewardConfig(use_amp=False))

    first_losses = trainer.train_epoch(dataloader)
    later_losses = []
    for _ in range(20):
        later_losses = trainer.train_epoch(dataloader)

    assert sum(later_losses) / len(later_losses) < sum(first_losses) / len(first_losses)

    accuracy = trainer.evaluate_ranking_accuracy(dataloader)
    assert 0.0 <= accuracy <= 1.0


def test_reward_trainer_save_checkpoint(tmp_path):
    model, _ = _tiny_reward_model_and_tokenizer()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    trainer = RewardTrainer(model, optimizer, torch.device("cpu"), RewardConfig(use_amp=False))

    ckpt_path = tmp_path / "reward.pt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()

    state_dict = torch.load(ckpt_path)
    assert "value_head.weight" in state_dict
