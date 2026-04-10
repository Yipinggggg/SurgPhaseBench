import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TMRNetHead(nn.Module):
    """
    Refactored, dimension-agnostic TMRNet head.
    Matches the logic in TemporalModel/TMRNet but removes hardcoded values.
    
    Args:
        short_dim: Feature dimension from the trainable encoder (S_t)
        memory_dim: Feature dimension from the prebuilt memory bank (L_t)
        num_classes: Number of output phase classes
        dropout: Dropout probability (default 0.2)
    """
    def __init__(self, short_dim=512, memory_dim=512, num_classes=7, dropout=0.2):
        super(TMRNetHead, self).__init__()
        
        # NLBlock logic
        self.linear1 = nn.Linear(short_dim, short_dim)
        self.linear2 = nn.Linear(memory_dim, short_dim)
        self.linear3 = nn.Linear(memory_dim, short_dim)
        self.linear4 = nn.Linear(short_dim, short_dim)
        self.layer_norm = nn.LayerNorm([1, short_dim])
        self.dropout_nl = nn.Dropout(dropout)

        # Multi-scale TimeConv logic (kernel 3, 5, 7)
        self.timeconv1 = nn.Conv1d(memory_dim, memory_dim, kernel_size=3, padding=1)
        self.timeconv2 = nn.Conv1d(memory_dim, memory_dim, kernel_size=5, padding=2)
        self.timeconv3 = nn.Conv1d(memory_dim, memory_dim, kernel_size=7, padding=3)
        self.maxpool_m = nn.MaxPool1d(2, stride=1)
        
        # Final prediction head
        self.fc_merge = nn.Linear(short_dim * 2, short_dim)
        self.classifier = nn.Linear(short_dim, num_classes)
        self.dropout_final = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                init.xavier_uniform_(m.weight)

    def _time_conv(self, memory_bank):
        """Processes the memory bank context sliding window.
        memory_bank: (B, T_mem, C_mem) -> (B, T_mem, C_mem)
        """
        x = memory_bank.transpose(1, 2) # (B, C, T)
        
        y1 = self.timeconv1(x)
        y2 = self.timeconv2(x)
        y3 = self.timeconv3(x)
        
        x_pad = F.pad(x, (1, 0), mode='constant', value=0)
        y4 = self.maxpool_m(x_pad)
        
        # Aggregation: max pool across scales + identity
        y = torch.max(torch.stack([x, y1, y2, y3, y4], dim=0), dim=0)[0]
        return y.transpose(1, 2) # (B, T, C)

    def forward(self, short_feat, memory_bank):
        """
        short_feat: (B, C_short) - Current frame feature
        memory_bank: (B, T_mem, C_mem) - Historical window
        """
        # 1. Process long term bank
        lt_bank = self._time_conv(memory_bank) # (B, T_mem, C_mem)
        
        # 2. NLBlock (Relational Reasoning)
        st_query = self.linear1(short_feat.unsqueeze(1)) # (B, 1, C_short)
        lt_key = self.linear2(lt_bank).transpose(1, 2)   # (B, C_short, T_mem)
        
        # Attention score
        attn = torch.matmul(st_query, lt_key) * ((1.0 / st_query.size(-1)) ** 0.5)
        attn = F.softmax(attn, dim=2) # (B, 1, T_mem)
        
        lt_val = self.linear3(lt_bank) # (B, T_mem, C_short)
        relational_delta = torch.matmul(attn, lt_val) # (B, 1, C_short)

        relational_delta = F.relu(self.layer_norm(relational_delta))
        relational_delta = self.dropout_nl(self.linear4(relational_delta))
        relational_delta = relational_delta.squeeze(1) # (B, C_short)

        # Original NLBlock returns St + SLL.
        relational_feat = short_feat + relational_delta
        
        # 3. Predict
        combined = torch.cat([short_feat, relational_feat], dim=1) # (B, 2*C_short)
        out = F.relu(self.dropout_final(self.fc_merge(combined)))
        logits = self.classifier(out)
        
        return logits
