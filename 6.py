import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

pairs = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("i am fine", "je vais bien"),
    ("thank you", "merci"),
    ("good night", "bonne nuit"),
]

tgt_start, tgt_end = "<sos>", "<eos>"

src_texts = [src.split() for src, _ in pairs]

tgt_texts = [
    [tgt_start] + tgt.split() + [tgt_end]
    for _, tgt in pairs
]

def build_vocab(sentences):
    vocab = {"<pad>": 0, "<unk>": 1}

    for sent in sentences:
        for w in sent:
            if w not in vocab:
                vocab[w] = len(vocab)

    return vocab

src_vocab = build_vocab(src_texts)
tgt_vocab = build_vocab(tgt_texts)

inv_tgt_vocab = {
    i: w for w, i in tgt_vocab.items()
}

def encode(sentence, vocab):
    return [
        vocab.get(w, vocab["<unk>"])
        for w in sentence
    ]

src_encoded = [
    torch.tensor(encode(s, src_vocab))
    for s in src_texts
]

tgt_encoded = [
    torch.tensor(encode(s, tgt_vocab))
    for s in tgt_texts
]

src_padded = pad_sequence(
    src_encoded,
    batch_first=True,
    padding_value=0
)

tgt_padded = pad_sequence(
    tgt_encoded,
    batch_first=True,
    padding_value=0
)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            batch_first=True
        )

    def forward(self, x):
        emb = self.embed(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, dec_hidden, enc_outputs):
        dec_hidden = dec_hidden.permute(1, 0, 2)

        score = self.V(
            torch.tanh(
                self.W1(enc_outputs) +
                self.W2(dec_hidden)
            )
        )

        attn_weights = torch.softmax(score, dim=1)

        context = torch.sum(
            attn_weights * enc_outputs,
            dim=1,
            keepdim=True
        )

        return context, attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            hidden_size + embed_dim,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

        self.attn = BahdanauAttention(hidden_size)

    def forward(self, x, hidden, cell, enc_outputs):
        x = self.embed(x)

        context, attn_weights = self.attn(
            hidden,
            enc_outputs
        )

        lstm_input = torch.cat(
            [x, context],
            dim=-1
        )

        outputs, (h, c) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        logits = self.fc(outputs)

        return logits, (h, c), attn_weights

embed_dim, hidden_size = 32, 64

encoder = Encoder(
    len(src_vocab),
    embed_dim,
    hidden_size
)

decoder = Decoder(
    len(tgt_vocab),
    embed_dim,
    hidden_size
)

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)

optimizer = optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()),
    lr=0.01
)

epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()

    enc_outs, (h, c) = encoder(src_padded)

    loss = 0

    dec_input = tgt_padded[:, 0].unsqueeze(1)

    for t in range(1, tgt_padded.size(1)):
        logits, (h, c), _ = decoder(
            dec_input,
            h,
            c,
            enc_outs
        )

        loss += criterion(
            logits.squeeze(1),
            tgt_padded[:, t]
        )

        dec_input = tgt_padded[:, t].unsqueeze(1)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss={loss.item():.4f}"
        )

def translate(sentence):
    with torch.no_grad():
        seq = torch.tensor([
            encode(sentence.split(), src_vocab)
        ])

        enc_outs, (h, c) = encoder(seq)

        dec_input = torch.tensor([
            [tgt_vocab[tgt_start]]
        ])

        result = []

        for _ in range(10):
            logits, (h, c), _ = decoder(
                dec_input,
                h,
                c,
                enc_outs
            )

            pred_id = logits.argmax(-1).item()

            if pred_id == tgt_vocab[tgt_end]:
                break

            result.append(
                inv_tgt_vocab.get(
                    pred_id,
                    "<unk>"
                )
            )

            dec_input = torch.tensor([
                [pred_id]
            ])

    return " ".join(result)

tests = [
    "hello",
    "thank you",
    "good night",
    "how are you"
]

for s in tests:
    print(f"{s} → {translate(s)}")
