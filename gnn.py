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

Passage à l'échelle (514 stations + plus d'un millier de tronçons HERE) : la matrice adaptative
n'est gardée que sous 1 000 noeuds ; le calendrier, commun à tous les noeuds, est passé une seule
fois par pas de temps au lieu d'être recopié sur chaque noeud. Les matrices amont / aval restent
denses jusqu'à 4 000 noeuds : sur CPU, le produit dense (MKL) est plus rapide que le produit creux
à cette taille ; au-delà elles passent en creux (CSR).
"""
import numpy as np
import torch
import torch.nn as nn


class ConvTemporelle(nn.Module):
    """Convolution causale de noyau k le long du temps, à porte (GLU), écrite comme une couche linéaire
    sur les k pas concaténés : sur CPU, c'est nettement plus rapide qu'une Conv2d."""

    def __init__(self, d, k=3):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(k * d, 2 * d)

    def forward(self, x):  # x [B, T, N, d]
        T = x.shape[1]
        xp = nn.functional.pad(x, (0, 0, 0, 0, self.k - 1, 0))  # causal : zéros avant le début
        y = self.lin(torch.cat([xp[:, i:i + T] for i in range(self.k)], -1))
        a, b = y.chunk(2, dim=-1)
        return torch.tanh(a) * torch.sigmoid(b)


class ConvGraphe(nn.Module):
    def __init__(self, d, n_supports, K=2):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(d * (1 + n_supports * K), d)

    def forward(self, x, supports):  # x [B, T, N, d]
        sorties = [x]
        B, T, N, d = x.shape
        for A in supports:
            h = x
            for _ in range(self.K):
                # un seul produit [N, N] @ [N, B*T*d] (une GEMM, ou un produit creux au-delà de 4 000 noeuds)
                h = (A @ h.permute(2, 0, 1, 3).reshape(N, -1)).reshape(N, B, T, d).permute(1, 2, 0, 3)
                sorties.append(h)
        return self.lin(torch.cat(sorties, dim=-1))


class STGNNLeger(nn.Module):
    def __init__(self, n_noeuds, n_feat, A_amont, A_aval, n_cal=0, adaptative=None, d=32, emb=8, n_blocs=2, K=2,
                 dropout=0.1):
        super().__init__()
        # hors du state_dict : on les reconstruit à partir du graphe au rechargement
        creux = (lambda a: a.to_sparse_csr()) if n_noeuds > 4000 else (lambda a: a)
        self.register_buffer("A_amont", creux(torch.tensor(A_amont)), persistent=False)
        self.register_buffer("A_aval", creux(torch.tensor(A_aval)), persistent=False)
        self.adaptative = n_noeuds <= 1000 if adaptative is None else adaptative
        self.e1 = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.plongement = nn.Parameter(torch.randn(n_noeuds, emb) * 0.1)
        self.entree = nn.Linear(n_feat + n_cal + emb, d)
        self.blocs_t = nn.ModuleList(ConvTemporelle(d) for _ in range(n_blocs))
        self.blocs_g = nn.ModuleList(ConvGraphe(d, 3 if self.adaptative else 2, K) for _ in range(n_blocs))
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for _ in range(n_blocs))
        self.drop = nn.Dropout(dropout)
        self.sortie = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def adjacence_adaptative(self):
        return torch.softmax(torch.relu(self.e1 @ self.e2.T), dim=1)

    def forward(self, x, cal=None):  # x [B, T, N, F], cal [B, T, Fc]  ->  [B, N] dans [0, 1]
        B, T, N, _ = x.shape
        parts = [x, self.plongement[None, None].expand(B, T, N, -1)]
        if cal is not None:
            parts.append(cal[:, :, None, :].expand(B, T, N, -1))
        h = self.entree(torch.cat(parts, -1))  # [B, T, N, d]
        supports = [self.A_amont, self.A_aval] + ([self.adjacence_adaptative()] if self.adaptative else [])
        for i, (bt, bg, bn) in enumerate(zip(self.blocs_t, self.blocs_g, self.norms)):
            r, h = h, bt(h)
            if i == len(self.blocs_t) - 1:  # seule la sortie au dernier pas sert : on n'y diffuse que lui
                r, h = r[:, -1:], h[:, -1:]
            h = bg(h, supports)
            h = self.drop(h) + r
            h = bn(h.reshape(-1, h.shape[-1])).reshape(h.shape)
        return torch.sigmoid(self.sortie(h[:, -1]).squeeze(-1))


def fenetres(X, t_idx, T_in):
    """X [T, N, F] -> [len(t_idx), T_in, N, F] (fenêtre se terminant à t inclus)."""
    idx = np.asarray(t_idx)[:, None] + np.arange(-T_in + 1, 1)[None, :]
    return X[idx]


def _tenseur(X):
    return torch.from_numpy(np.ascontiguousarray(np.nan_to_num(X), dtype=np.float32))


def entrainer_gnn(X, Y, idx_train, idx_val, A_amont, A_aval, T_in, n_stations, cal=None,
                  epochs=30, lr=2e-3, batch=64, patience=6, graine=0, verbose=True, appareil=None):
    """X [T, N, F] variables par noeud ; cal [T, Fc] variables communes (calendrier) ; Y [T, S] occupation
    (NaN = inconnue) ; les S stations sont les S premiers noeuds. Perte : erreur absolue sur les stations
    observées uniquement. Calcul sur GPU s'il y en a un."""
    torch.manual_seed(graine)
    rng = np.random.default_rng(graine)
    dev = torch.device(appareil or ("cuda" if torch.cuda.is_available() else "cpu"))
    Xt = _tenseur(X)
    Ct = _tenseur(cal) if cal is not None else None
    Yt = torch.tensor(np.nan_to_num(Y, nan=-1), dtype=torch.float32)
    modele = STGNNLeger(X.shape[1], X.shape[2], A_amont, A_aval, n_cal=0 if cal is None else cal.shape[1]).to(dev)
    opt = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)

    def perte(ids):
        xb = fenetres(Xt, ids, T_in).to(dev)
        cb = fenetres(Ct, ids, T_in).to(dev) if Ct is not None else None
        yb = Yt[ids].to(dev)
        p = modele(xb, cb)[:, :n_stations]
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
            lv = np.mean([perte(idx_val[i:i + 256]).item() for i in range(0, len(idx_val), 256)])
        if verbose:
            print(f"  GNN époque {ep + 1:02d}  MAE validation {lv:.4f}", flush=True)
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
def predire_gnn(modele, X, t_idx, T_in, n_stations, cal=None):
    dev = next(modele.parameters()).device
    Xt = _tenseur(X)
    Ct = _tenseur(cal) if cal is not None else None
    out = []
    for i in range(0, len(t_idx), 256):
        ids = t_idx[i:i + 256]
        cb = fenetres(Ct, ids, T_in).to(dev) if Ct is not None else None
        out.append(modele(fenetres(Xt, ids, T_in).to(dev), cb)[:, :n_stations].cpu().numpy())
    return np.concatenate(out)
