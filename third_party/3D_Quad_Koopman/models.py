# models.py
import torch
import torch.nn as nn
from typing import Type

def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    depth: int,
    activation_fn: Type[nn.Module] | str = nn.ReLU
) -> nn.Sequential:
    """Helper function to create a multi-layer perceptron."""
    if isinstance(activation_fn, str):
        # Local import to avoid circular dependency
        from utils import get_activation_fn
        activation_fn = get_activation_fn(activation_fn)

    if depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))
    
    layers = [nn.Linear(input_dim, hidden_width), activation_fn()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_width, hidden_width), activation_fn()])
    layers.append(nn.Linear(hidden_width, output_dim))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_width: int, depth: int, activation_fn: Type[nn.Module] | str):
        super().__init__()
        self.net = create_mlp(input_dim, latent_dim, hidden_width, depth, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_width: int, depth: int, activation_fn: Type[nn.Module] | str):
        super().__init__()
        self.net = create_mlp(latent_dim, output_dim, hidden_width, depth, activation_fn)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(z) # Directly output the 7D state vector

class DeepKoopman(nn.Module):
    """
    The main model combining the encoder, decoder, and linear dynamics.
    """
    def __init__(self, state_dim: int, control_dim: int, latent_dim: int, hidden_width: int, depth: int, activation_fn: Type[nn.Module] | str):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim, hidden_width, depth, activation_fn)
        self.decoder = Decoder(latent_dim, state_dim, hidden_width, depth, activation_fn)
        
        # Learnable A and B matrices for the linear dynamics
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        x_k: [B, STATE_DIM] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, STATE_DIM] - Sequence of M target next states
        """

        # Get shapes
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]
        latent_dim = self.A.in_features

        # --- 1. Encode all ground truth states ---
        # x_k is [B, 12], x_next_seq is [B, M, 12]
        z_k = self.encoder(x_k)

        # Reshape for batch encoding: [B, M, 12] -> [B*M, 12]
        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.encoder(x_next_seq_flat)
        # Reshape back: [B*M, N] -> [B, M, N]
        z_target_seq = z_target_seq_flat.view(-1, M, latent_dim)

        # --- 2. Predict latent sequence (A, B rollout) ---
        z_pred_list = []
        z_current_pred = z_k 

        for i in range(M):
            u_i = u_seq[:, i, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        # Reshape for efficient decoding: [B, M, N] -> [B*M, N]
        z_pred_seq_flat = z_pred_seq.view(-1, latent_dim)

        # --- 3. Decode all latent tensors ---
        # Decode E(x_k) for recon loss 1a
        x_k_hat = self.decoder(z_k) 

        # Decode E(x_next_seq) for recon loss 1a
        x_target_seq_hat_flat = self.decoder(z_target_seq_flat)
        x_target_seq_hat = x_target_seq_hat_flat.view(-1, M, state_dim)

        # Decode the A,B rollout for dynamics loss 2b
        x_pred_seq_hat_flat = self.decoder(z_pred_seq_flat)
        x_pred_seq_hat = x_pred_seq_hat_flat.view(-1, M, state_dim)

        # --- 4. Return all tensors for loss calculation ---
        return (
            x_k_hat,              # D(E(x_k))
            x_target_seq_hat,     # D(E(x_next_seq))
            z_pred_seq,           # A,B rollout
            x_pred_seq_hat,       # D(A,B rollout)
            z_target_seq          # E(x_next_seq)
        )


class DeepKoopmanNoDec(nn.Module):
    """
    Decoder-free Koopman model with lifted state z = [x, enc(x)].
    """
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        embedding_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] | str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = state_dim + embedding_dim

        self.encoder = Encoder(state_dim, embedding_dim, hidden_width, depth, activation_fn)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(control_dim, self.latent_dim, bias=False)

    def lift_state(self, x: torch.Tensor) -> torch.Tensor:
        """Lift the state into Koopman coordinates z = [x, enc(x)]."""
        return torch.cat([x, self.encoder(x)], dim=-1)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        x_k: [B, STATE_DIM] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, STATE_DIM] - Sequence of M target next states
        """
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]

        z_k = self.lift_state(x_k)

        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.lift_state(x_next_seq_flat)
        z_target_seq = z_target_seq_flat.view(-1, M, self.latent_dim)

        z_pred_list = []
        z_current_pred = z_k
        for i in range(M):
            u_i = u_seq[:, i, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        x_pred_seq = z_pred_seq[..., :self.state_dim]

        return z_pred_seq, x_pred_seq, z_target_seq


class DeepKoopmanLinDec(nn.Module):
    """
    Koopman model with nonlinear encoder and linear decoder x = C z.
    """
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] | str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(state_dim, latent_dim, hidden_width, depth, activation_fn)
        self.C = nn.Linear(latent_dim, state_dim, bias=True)
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.C(z)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        x_k: [B, STATE_DIM] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, STATE_DIM] - Sequence of M target next states
        """
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]

        z_k = self.encoder(x_k)

        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.encoder(x_next_seq_flat)
        z_target_seq = z_target_seq_flat.view(-1, M, self.latent_dim)

        z_pred_list = []
        z_current_pred = z_k
        for i in range(M):
            u_i = u_seq[:, i, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        x_pred_seq = self.decode(z_pred_seq)

        return z_pred_seq, x_pred_seq, z_target_seq


class DeepKoopmanSplit(nn.Module):
    """
    Split-branch Koopman model with separate position/rest encoders and decoders.
    """
    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim_pos: int,
        latent_dim_rest: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] | str,
    ):
        super().__init__()
        if state_dim != 12:
            raise ValueError("DeepKoopmanSplit expects state_dim=12 (3 pos + 9 rest).")
        self.latent_dim_pos = latent_dim_pos
        self.latent_dim_rest = latent_dim_rest
        latent_dim = latent_dim_pos + latent_dim_rest

        self.encoder_pos = Encoder(3, latent_dim_pos, hidden_width, depth, activation_fn)
        self.encoder_rest = Encoder(9, latent_dim_rest, hidden_width, depth, activation_fn)
        self.decoder_pos = Decoder(latent_dim_pos, 3, hidden_width, depth, activation_fn)
        self.decoder_rest = Decoder(latent_dim_rest, 9, hidden_width, depth, activation_fn)

        # Learnable A and B matrices for the linear dynamics
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        x_k: [B, 12] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, 12] - Sequence of M target next states
        """
        M = u_seq.shape[1]
        latent_dim = self.A.in_features

        x_k_pos = x_k[:, :3]
        x_k_rest = x_k[:, 3:]

        z_k_pos = self.encoder_pos(x_k_pos)
        z_k_rest = self.encoder_rest(x_k_rest)
        z_k = torch.cat([z_k_pos, z_k_rest], dim=-1)

        x_next_seq_flat = x_next_seq.view(-1, 12)
        x_next_pos_flat = x_next_seq_flat[:, :3]
        x_next_rest_flat = x_next_seq_flat[:, 3:]

        z_target_pos_flat = self.encoder_pos(x_next_pos_flat)
        z_target_rest_flat = self.encoder_rest(x_next_rest_flat)
        z_target_seq_flat = torch.cat([z_target_pos_flat, z_target_rest_flat], dim=-1)
        z_target_seq = z_target_seq_flat.view(-1, M, latent_dim)

        # --- Predict latent sequence (A, B rollout) ---
        z_pred_list = []
        z_current_pred = z_k
        for i in range(M):
            u_i = u_seq[:, i, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred
        z_pred_seq = torch.stack(z_pred_list, dim=1)
        z_pred_seq_flat = z_pred_seq.view(-1, latent_dim)

        # --- Decode all latent tensors ---
        x_k_hat_pos = self.decoder_pos(z_k_pos)
        x_k_hat_rest = self.decoder_rest(z_k_rest)
        x_k_hat = torch.cat([x_k_hat_pos, x_k_hat_rest], dim=-1)

        z_target_pos_flat = z_target_seq_flat[:, :self.latent_dim_pos]
        z_target_rest_flat = z_target_seq_flat[:, self.latent_dim_pos:]
        x_target_pos_hat_flat = self.decoder_pos(z_target_pos_flat)
        x_target_rest_hat_flat = self.decoder_rest(z_target_rest_flat)
        x_target_seq_hat_flat = torch.cat([x_target_pos_hat_flat, x_target_rest_hat_flat], dim=-1)
        x_target_seq_hat = x_target_seq_hat_flat.view(-1, M, 12)

        z_pred_pos_flat = z_pred_seq_flat[:, :self.latent_dim_pos]
        z_pred_rest_flat = z_pred_seq_flat[:, self.latent_dim_pos:]
        x_pred_pos_hat_flat = self.decoder_pos(z_pred_pos_flat)
        x_pred_rest_hat_flat = self.decoder_rest(z_pred_rest_flat)
        x_pred_seq_hat_flat = torch.cat([x_pred_pos_hat_flat, x_pred_rest_hat_flat], dim=-1)
        x_pred_seq_hat = x_pred_seq_hat_flat.view(-1, M, 12)

        return (
            x_k_hat,          # D_pos(E_pos(x_k_pos)) + D_rest(E_rest(x_k_rest))
            x_target_seq_hat, # D_pos(E_pos(x_next_pos)) + D_rest(E_rest(x_next_rest))
            z_pred_seq,       # A,B rollout
            x_pred_seq_hat,   # D_pos(z_pred_pos) + D_rest(z_pred_rest)
            z_target_seq      # E_pos(x_next_pos) + E_rest(x_next_rest)
        )


class DeepKoopmanInt(nn.Module):
    """
    Integration-based Deep Koopman model.
    Instead of predicting z_{k+1} = A*z_k + B*u_k directly,
    this model predicts dz = A*z_k + B*u_k and integrates: z_{k+1} = z_k + dz
    This is similar to a ResNet skip connection for dynamics.
    """
    def __init__(self, state_dim: int, control_dim: int, latent_dim: int, hidden_width: int, depth: int, activation_fn: Type[nn.Module] | str):
        super().__init__()
        self.encoder = Encoder(state_dim, latent_dim, hidden_width, depth, activation_fn)
        self.decoder = Decoder(latent_dim, state_dim, hidden_width, depth, activation_fn)
        
        # Learnable A and B matrices for predicting dz (change in latent state)
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        x_k: [B, STATE_DIM] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, STATE_DIM] - Sequence of M target next states
        """

        # Get shapes
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]
        latent_dim = self.A.in_features

        # --- 1. Encode all ground truth states ---
        z_k = self.encoder(x_k)

        # Reshape for batch encoding: [B, M, state_dim] -> [B*M, state_dim]
        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.encoder(x_next_seq_flat)
        # Reshape back: [B*M, N] -> [B, M, N]
        z_target_seq = z_target_seq_flat.view(-1, M, latent_dim)

        # --- 2. Predict latent sequence using INTEGRATION (ResNet-style) ---
        # dz = A*z_k + B*u_k, then z_{k+1} = z_k + dz
        z_pred_list = []
        z_current_pred = z_k 

        for i in range(M):
            u_i = u_seq[:, i, :]
            dz = self.A(z_current_pred) + self.B(u_i)  # Predict the change
            z_next_pred = z_current_pred + dz          # Integrate (skip connection)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        # Reshape for efficient decoding: [B, M, N] -> [B*M, N]
        z_pred_seq_flat = z_pred_seq.view(-1, latent_dim)

        # --- 3. Decode all latent tensors ---
        # Decode E(x_k) for recon loss
        x_k_hat = self.decoder(z_k) 

        # Decode E(x_next_seq) for recon loss
        x_target_seq_hat_flat = self.decoder(z_target_seq_flat)
        x_target_seq_hat = x_target_seq_hat_flat.view(-1, M, state_dim)

        # Decode the integrated A,B rollout for dynamics loss
        x_pred_seq_hat_flat = self.decoder(z_pred_seq_flat)
        x_pred_seq_hat = x_pred_seq_hat_flat.view(-1, M, state_dim)

        # --- 4. Return all tensors for loss calculation ---
        return (
            x_k_hat,              # D(E(x_k))
            x_target_seq_hat,     # D(E(x_next_seq))
            z_pred_seq,           # Integrated A,B rollout
            x_pred_seq_hat,       # D(integrated A,B rollout)
            z_target_seq          # E(x_next_seq)
        )

class DeepKoopmanObserver(DeepKoopmanInt):
    """
    Integration-based Deep Koopman model with an additional Linear Observer (C).
    The matrix C maps the latent state z to the position [x, y, z].
    [x, y, z]^T = C * z
    """
    def __init__(self, state_dim: int, control_dim: int, latent_dim: int, hidden_width: int, depth: int, activation_fn: Type[nn.Module] | str):
        super().__init__(state_dim, control_dim, latent_dim, hidden_width, depth, activation_fn)
        # C maps latent_dim -> 3 (x, y, z)
        self.C = nn.Linear(latent_dim, 3, bias=False)

    def forward(self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor) -> tuple:
        """
        Extends DeepKoopmanInt forward to also return position predictions from C.
        """
        # Get shapes
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]
        latent_dim = self.A.in_features

        # --- 1. Encode all ground truth states ---
        z_k = self.encoder(x_k)

        # Reshape for batch encoding
        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.encoder(x_next_seq_flat)
        z_target_seq = z_target_seq_flat.view(-1, M, latent_dim)

        # --- 2. Predict latent sequence using INTEGRATION ---
        z_pred_list = []
        z_current_pred = z_k 

        for i in range(M):
            u_i = u_seq[:, i, :]
            dz = self.A(z_current_pred) + self.B(u_i)
            z_next_pred = z_current_pred + dz
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        z_pred_seq_flat = z_pred_seq.view(-1, latent_dim)

        # --- 3. Decode all latent tensors (Decoder D) ---
        x_k_hat = self.decoder(z_k) 
        x_target_seq_hat_flat = self.decoder(z_target_seq_flat)
        x_target_seq_hat = x_target_seq_hat_flat.view(-1, M, state_dim)
        x_pred_seq_hat_flat = self.decoder(z_pred_seq_flat)
        x_pred_seq_hat = x_pred_seq_hat_flat.view(-1, M, state_dim)

        # --- 4. Linear Observer Predictions (Matrix C) ---
        # Predict position from z_k
        pos_k_hat = self.C(z_k)

        # Predict position from z_target_seq (ground truth latents)
        pos_target_seq_hat_flat = self.C(z_target_seq_flat)
        pos_target_seq_hat = pos_target_seq_hat_flat.view(-1, M, 3)

        # Predict position from z_pred_seq (rollout latents)
        pos_pred_seq_hat_flat = self.C(z_pred_seq_flat)
        pos_pred_seq_hat = pos_pred_seq_hat_flat.view(-1, M, 3)

        return (
            x_k_hat,              # D(E(x_k))
            x_target_seq_hat,     # D(E(x_next_seq))
            z_pred_seq,           # Integrated A,B rollout
            x_pred_seq_hat,       # D(integrated A,B rollout)
            z_target_seq,         # E(x_next_seq)
            pos_k_hat,            # C(E(x_k))
            pos_target_seq_hat,   # C(E(x_next_seq))
            pos_pred_seq_hat,     # C(integrated A,B rollout)
            z_k                   # E(x_k) - returned for convenience if needed
        )
