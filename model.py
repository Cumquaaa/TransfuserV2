import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Where to put this?
def transfuser_mask(b, h, q_idx, kv_idx):
    # Causal mask for text tokens

    # Image tokens between <BOI> and <EOI>

    # <EOI> token (optional: depends on if it should attend to image tokens or not)

    return False  # Default case (not strictly necessary, but for clarity)

# Create the block mask using the custom function
block_mask = create_block_mask(transfuser_mask, B=None, H=None, Q_LEN=0, KV_LEN=0)


class TransfuserAttention(nn.Module):
    def __init__(self, config, layer_idx):
        """
        Custom attention class for the Transfuser model.

        Args:
            config (TransfuerConfig): Configuration class for the model.
            layer_idx (int): Index of the layer to which this attention belongs.
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx

        # Define linear layers for query, key, value, and output projection
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, token_type_ids=None):
        """
        Forward pass for the Transfuser attention mechanism.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (torch.Tensor, optional): Mask to avoid attention on padding tokens.
            token_type_ids (torch.Tensor, optional): Tensor indicating the type of input tokens (text/image).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_size).
        """
        batch_size, seq_length, hidden_size = hidden_states.size()

        # Project hidden states to query, key, value
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Split into multiple heads
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Create the block mask using the custom `transfuser_mask` function
        block_mask = create_block_mask(transfuser_mask, B=batch_size, H=self.num_heads, Q_LEN=seq_length, KV_LEN=seq_length)

        # Apply the flex_attention function
        attn_output = flex_attention(query, key, value, block_mask=block_mask)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        output = self.out_proj(attn_output)

        # Optional: Apply layer normalization
        if self.config.use_layer_norm:
            output = self.layer_norm(output)

        # Should return attn_output, attn_weights, past_key_value?
        return output


class TransfuerConfig(LlamaConfig):
    def __init__(self, token_type=None, **kwargs):
        """
        Custom configuration class to add support for a tensor indicating input token types (text/image).

        Args:
            token_type (torch.Tensor, optional): A tensor indicating the type of input tokens (text/image).
            **kwargs: Additional arguments passed to the base LlamaConfig.
        """
        super().__init__(**kwargs)
        self.token_type = token_type
        
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            dict: Dictionary representation of the configuration.
        """
        output = super().to_dict()
        # Convert the token_type tensor to a list if it is not None
        output["token_type"] = self.token_type.tolist() if self.token_type is not None else None
        return output


class TransfuserForCausalLM(LlamaForCausalLM):
    def __init__(self, config: TransfuerConfig):
        """
        Transfuser model class inheriting from LlamaForCausalLM, using TransfuerConfig.

        Args:
            config (TransfuerConfig): Configuration class for the model.
        """
        super().__init__(config)
        self.config = config
        
        for i, layer in enumerate(self.model.layers):
            layer.self_attn = TransfuserAttention(config, i)
