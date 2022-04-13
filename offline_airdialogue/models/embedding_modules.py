import math
import torch.nn as nn
import torch.nn.functional as F

class FlightTableEmbedder(nn.Module):
    def __init__(self, emb_spec, dim, n_layers):
        super(FlightTableEmbedder, self).__init__()
        self.flight_embedder = EmbeddingCombiner(emb_spec, dim)
        self.flight_mlp = MLP(dim, [dim]*n_layers, dim)

    def forward(self, flight_tables):
        return self.flight_mlp(self.flight_embedder(flight_tables))

class EmbeddingCombiner(nn.Module):
    # embedding_spec is a dict: k = embedding_name, v = num_embeddings
    def __init__(self, emb_spec, emb_dim):
        super(EmbeddingCombiner, self).__init__()
        self.embeddings = nn.ModuleDict({k: nn.Embedding(num_embeddings=num_embs, embedding_dim=emb_dim)
                                         for k, num_embs in emb_spec.items()})

    # idx_dict should be int tensor values, with keys matching the emb_spec keys
    def forward(self, idx_dict, check_embeddings=True):
        if check_embeddings:
            assert set(idx_dict.keys()) == set(self.embeddings.keys())
        final_emb = 0.0
        num_keys = len(self.embeddings)
        for k in list(self.embeddings.keys()):
            final_emb += self.embeddings[k](idx_dict[k]) / math.sqrt(num_keys)
        return final_emb

class MLP(nn.Module):
    # hidden dims is a list
    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLP, self).__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers = []
        for i in range(len(dims-1)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
        self.layer_modules = nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layer_modules)-1):
            x = F.gelu(self.layer_modules[i](x))
        return self.layer_modules[-1](x)