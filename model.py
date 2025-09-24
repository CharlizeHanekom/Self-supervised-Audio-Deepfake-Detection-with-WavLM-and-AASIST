import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig

class WavLMFeatureExtractor(nn.Module):
    """
    WavLM feature extractor with optional fine-tuning.
    It extracts features from the last hidden state.
    """
    def __init__(self, model_path: str = "models/wavlm-base", freeze: bool = True):
        super().__init__()
        print(f"Loading WavLM from: {model_path}")
        self.config = WavLMConfig.from_pretrained(model_path)
        self.wavlm = WavLMModel.from_pretrained(model_path)
        self.wavlm.feature_extractor._requires_grad = False

        if freeze:
            print("Freezing WavLM parameters.")
            for param in self.wavlm.parameters():
                param.requires_grad = False
        else:
            print("WavLM parameters will be fine-tuned.")

        self.output_dim = self.config.hidden_size

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from raw audio waveforms.
        Args:
            waveforms: Input audio tensor of shape (batch, seq_len)
        Returns:
            features: Extracted features of shape (batch, seq_len', hidden_size)
        """
        # WavLM expects a 1D waveform tensor
        if waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)

        # The model handles normalization internally
        outputs = self.wavlm(waveforms, output_hidden_states=True)
        
        # As per the proposal, using the second-to-last layer's features
        # The hidden states include the embedding layer + all transformer layers
        # So the second to last is the one before the final hidden state
        if len(outputs.hidden_states) > 1:
            # We take the output of the second-to-last transformer layer
            return outputs.hidden_states[-2]
        else:
            # Fallback to last hidden state if only one is available
            return outputs.last_hidden_state


class AASIST(nn.Module):
    """
    AASIST-inspired backend for audio spoofing detection.
    This is a simplified version focusing on core concepts:
    convolutional blocks for local feature extraction and an attention
    mechanism for global context aggregation.
    """
    def __init__(self, input_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )

        # Attentive Statistics Pooling
        self.asp_attention = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # FCNN Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), # Mean and Std Dev are concatenated
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        Returns:
            predictions: Output logits of shape (batch, 1)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x_conv = self.conv_block(x)

        # Attentive Statistics Pooling
        w = self.asp_attention(x_conv)
        mu = torch.sum(x_conv * w, dim=2)
        sg = torch.sqrt(torch.sum((x_conv**2) * w, dim=2) - mu**2 + 1e-9)
        x_pooled = torch.cat((mu, sg), dim=1)

        return self.classifier(x_pooled)


class WavLM_AASIST_Model(nn.Module):
    """
    Combined WavLM + AASIST model for audio deepfake detection.
    """
    def __init__(self, model_path: str = "models/wavlm-base", freeze_wavlm: bool = True):
        super().__init__()
        self.feature_extractor = WavLMFeatureExtractor(model_path, freeze_wavlm)
        self.backend = AASIST(input_dim=self.feature_extractor.output_dim)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(waveforms)
        predictions = self.backend(features)
        return predictions.squeeze(1)