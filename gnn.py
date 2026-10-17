"""
GNN spatio-temporel léger, dans l'esprit de STGCN / Graph WaveNet :
  convolution temporelle causale  ->  convolution de graphe par diffusion  ->  (x2)  ->  sortie par noeud.

Comment le réseau "voit" le réseau routier :
  - chaque noeud (station ou compteur) a sa propre série de variables sur une fenêtre de T_in pas ;
  - la convolution temporelle apprend les retards (une vague de trafic vue il y a 2 h au compteur) ;
  - la convolution de graphe mélange l'information entre noeuds selon les matrices A_amont / A_aval
    construites à partir des temps de parcours (graphe.py) ;
  - une matrice d'adjacence ADAPTATIVE (apprise, comme Graph WaveNet) corrige ce que le graphe
    théorique ne sait pas : report entre aires, compteur mal rattaché, etc. ;
  - un plongement (embedding) par noeud apprend les spécificités de chaque station (taille, attractivité).
"""
import numpy as np
import torch
import torch.nn as nn


class ConvTemporelle(nn.Module):
    def __init__(self, d, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(d, 2 * d, kernel_size=(1, k), padding=(0, k - 1))

    def forward(self, x):  # x [B, d, N, T]
        y = self.conv(x)[..., : x.shape[-1]]  # causal : on coupe ce qui viendrait du futur
        a, b = y.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class ConvGraphe(nn.Module):
    def __init__(self, d, n_supports, K=2):
        super().__init__()
        self.K = K
        self.mlp = nn.Conv2d(d * (1 + n_supports * K), d, kernel_size=1)

    def forward(self, x, supports):  # x [B, d, N, T]
        sorties = [x]
        for A in supports:
            h = x
            for _ in range(self.K):
                h = torch.einsum("ij,bdjt->bdit", A, h)
                sorties.append(h)
        return self.mlp(torch.cat(sorties, dim=1))


class STGNNLeger(nn.Module):
    def __init__(self, n_noeuds, n_feat, A_amont, A_aval, d=32, emb=8, n_blocs=2, K=2, dropout=0.1):
        super().__init__()
        self.register_buffer("A_amont", torch.tensor(A_amont))
        self.register_buffer("A_aval", torch.tensor(A_aval))
        self.e1 = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.plongement = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.entree = nn.Conv2d(n_feat + emb, d, kernel_size=1)
        self.blocs_t = nn.ModuleList(ConvTemporelle(d) for _ in range(n_blocs))
        self.blocs_g = nn.ModuleList(ConvGraphe(d, 3, K) for _ in range(n_blocs))
        self.norms = nn.ModuleList(nn.BatchNorm2d(d) for _ in range(n_blocs))
        self.drop = nn.Dropout(dropout)
        self.sortie = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def adjacence_adaptative(self):
        return torch.softmax(torch.relu(self.e1 @ self.e2.T), dim=1)

    def forward(self, x):  # x [B, T, N, F]  ->  [B, N] dans [0, 1]
        B, T, N, _ = x.shape
        emb = self.plongement[None, None].expand(B, T, N, -1)
        h = self.entree(torch.cat([x, emb], -1).permute(0, 3, 2, 1))  # [B, d, N, T]
        supports = [self.A_amont, self.A_aval, self.adjacence_adaptative()]
        for bt, bg, bn in zip(self.blocs_t, self.blocs_g, self.norms):
            r = h
            h = bg(bt(h), supports)
            h = bn(self.drop(h) + r)
        return torch.sigmoid(self.sortie(h[..., -1].transpose(1, 2)).squeeze(-1))


def fenetres(X, t_idx, T_in):
    """X [T, N, F] -> [len(t_idx), T_in, N, F] (fenêtre se terminant à t inclus)."""
    idx = np.asarray(t_idx)[:, None] + np.arange(-T_in + 1, 1)[None, :]
    return X[idx]


def entrainer_gnn(X, Y, idx_train, idx_val, A_amont, A_aval, T_in, n_stations,
                  epochs=30, lr=2e-3, batch=64, patience=6, graine=0, verbose=True):
    """X [T, N, F] variables par noeud ; Y [T, S] occupation (NaN = inconnue) ; les S stations sont
    les S premiers noeuds. Perte : erreur absolue sur les stations observées uniquement."""
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    Xt = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
    Yt = torch.tensor(np.nan_to_num(Y, nan=-1), dtype=torch.float32)
    modele = STGNNLeger(X.shape[1], X.shape[2], A_amont, A_aval)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)

    def perte(ids):
        xb = fenetres(Xt, ids, T_in)
        yb = Yt[ids]
        p = modele(xb)[:, :n_stations]
        m = yb >= 0
        return (torch.abs(p - yb) * m).sum() / m.sum().clamp(min=1)

    meilleur, etat, attente = np.inf, None, 0
    for ep in range(epochs):
        modele.train()
        ordre = rng.permutation(idx_train)
        for i in range(0, len(ordre), batch):
            opt.zero_grad()
            l = perte(ordre[i:i + batch])
            l.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 5.0)
            opt.step()
        modele.eval()
        with torch.no_grad():
            lv = np.mean([perte(idx_val[i:i + 512]).item() for i in range(0, len(idx_val), 512)])
        if verbose:
            print(f"  GNN époque {ep + 1:02d}  MAE validation {lv:.4f}")
        if lv < meilleur - 1e-4:
            meilleur, attente = lv, 0
            etat = {k: v.clone() for k, v in modele.state_dict().items()}
        else:
            attente += 1
            if attente >= patience:
                break
    modele.load_state_dict(etat)
    modele.eval()
    return modele


@torch.no_grad()
def predire_gnn(modele, X, t_idx, T_in, n_stations):
    Xt = torch.tensor(np.nan_to_num(X), dtype=torch.float32)
    out = [modele(fenetres(Xt, t_idx[i:i + 512], T_in))[:, :n_stations].numpy() for i in range(0, len(t_idx), 512)]
    return np.concatenate(out)
