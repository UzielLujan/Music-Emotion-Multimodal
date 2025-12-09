
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioNetwork(nn.Module):
    # 1. AGREGAMOS 'audio_1d_dim' AQUÍ como parámetro (puedes poner un valor por defecto, ej: 34 o 26)
    def __init__(self, num_classes=2, audio_1d_dim=34):
        super(AudioNetwork, self).__init__()
        
        # --- 1. RAMA VISUAL (CNN - Espectrograma) ---
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # --- 2. RAMA TEMPORAL (GRU + Atención) ---
        self.rnn_input_size = 64 * 8
        self.rnn_hidden_size = 128
        
        self.gru = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.rnn_hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.attention_layer = nn.Linear(self.rnn_hidden_size * 2, 1)

        # --- 3. RAMA DE DATOS 1D (Metadatos) ---
        # 2. AQUÍ USAMOS LA VARIABLE en lugar de un número fijo
        self.fc_1d = nn.Sequential(
            nn.Linear(audio_1d_dim, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # --- 4. FUSIÓN Y CLASIFICACIÓN ---
        self.fc_final = nn.Sequential(
            nn.Linear((self.rnn_hidden_size * 2) + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_2d, x_1d):
        # --- A. PROCESAMIENTO CNN ---
        x = self.pool1(F.relu(self.bn1(self.conv1(x_2d))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # --- B. PREPARACIÓN PARA GRU ---
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, C, H = x.size()
        x = x.reshape(batch_size, time_steps, C * H)

        # --- C. GRU ---
        gru_out, _ = self.gru(x)

        # --- D. MECANISMO DE ATENCIÓN ---
        attn_weights = self.attention_layer(gru_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)

        # --- E. PROCESAMIENTO 1D ---
        out_1d = self.fc_1d(x_1d)

        # --- F. FUSIÓN ---
        combined = torch.cat((context_vector, out_1d), dim=1)
        
        # --- G. SALIDA FINAL ---
        out = self.fc_final(combined)
        
        return out


# ==========================================
# 📝 MODELO DE TEXTO (Embeddings + TF-IDF Chi2)
# ==========================================

class TextNetwork(nn.Module):
    # CAMBIO IMPORTANTE: Agregamos text_1d_dim al constructor
    def __init__(self, num_classes=2, text_1d_dim=500):
        super(TextNetwork, self).__init__()
        
        # --- 1. RAMA TEXTUAL (CNN 1D - Embeddings) ---
        # Entrada: (Batch, Channels=768, Seq=256)
        
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.pool1 = nn.MaxPool1d(2) 
        
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(2)
        
        # --- 2. RAMA TEMPORAL (GRU + Atención) ---
        self.rnn_input_size = 64 
        self.rnn_hidden_size = 128
        
        self.gru = nn.GRU(
            input_size=self.rnn_input_size, 
            hidden_size=self.rnn_hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.attention_layer = nn.Linear(self.rnn_hidden_size * 2, 1)

        # --- 3. RAMA DE DATOS 1D (Metadatos + TF-IDF SelectKBest) ---
        # CAMBIO IMPORTANTE: Usamos la variable text_1d_dim en lugar del fijo '26'
        self.fc_1d = nn.Sequential(
            nn.Linear(text_1d_dim, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # --- 4. FUSIÓN Y CLASIFICACIÓN ---
        self.fc_final = nn.Sequential(
            nn.Linear((self.rnn_hidden_size * 2) + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_text, x_1d):
        # x_text shape: (Batch, 768, 256) gracias al Dataset corregido
        
        # --- A. PROCESAMIENTO CNN ---
        x = self.pool1(F.relu(self.bn1(self.conv1(x_text))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # x shape: (Batch, 64, 16)

        # --- B. PREPARACIÓN PARA GRU ---
        # GRU necesita: (Batch, Time, Features)
        x = x.permute(0, 2, 1) # (Batch, 16, 64)

        # --- C. GRU Y ATENCIÓN ---
        gru_out, _ = self.gru(x)
        
        attn_weights = self.attention_layer(gru_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)

        # --- D. PROCESAMIENTO METADATOS ---
        out_1d = self.fc_1d(x_1d)

        # --- E. FUSIÓN ---
        combined = torch.cat((context_vector, out_1d), dim=1)
        
        # --- F. SALIDA FINAL ---
        out = self.fc_final(combined)
        
        return out