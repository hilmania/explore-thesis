from torch.utils.data import DataLoader

# Inisialisasi dataset dan dataloader
dataset = EEGDatasetH5('data/eeg_file.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = EEGNet(input_channels=20)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop 1 epoch (contoh)
for batch_eeg, batch_labels in dataloader:
    outputs = model(batch_eeg)
    loss = criterion(outputs, batch_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Loss:", loss.item())
