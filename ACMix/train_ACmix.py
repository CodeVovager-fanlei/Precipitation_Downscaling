import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.util import BernoulliGammaLoss
from utils.data_process import getdata, split_data, standardize_dataset, train_val_split, prepare_data
from ACmix import ACMixModel

tensor_x, tensor_y = getdata()
val_ratio = 0.2
input_shape = tensor_x.shape[1:]
output_shape = tensor_y.shape[1:]
model = ACMixModel(input_shape, output_shape)

best_val_loss = float('inf')
early_stop = None

num_epochs = 2000
learning_rate = 0.0001
criterion = BernoulliGammaLoss("conv")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i in range(5):
    print(f'第{i}折')
    start_time = '2001-01-01'
    cut_time = '2010-12-31'
    train_x, test_x, train_y, test_y = split_data(start_time, cut_time, tensor_x, tensor_y)
    # 标准化处理
    train_x_scaled, test_x_scaled = standardize_dataset(train_x, test_x)
    # 划分训练集和验证集
    train_x, val_x, train_y, val_y = train_val_split(train_x_scaled, train_y, ratio=val_ratio, k_fold=True, index=i)
    train_dataset, val_dataset, test_dataset = prepare_data(train_x, train_y, val_x, val_y, test_x_scaled, test_y)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Early stopping parameters
    early_stopping_patience = 50
    epochs_no_improve = 0
    early_stop = False

    # Training loop with early stopping
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping")
            break

        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(targets, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                loss = criterion(val_targets, val_outputs)
                val_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} ')

        # Check early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'model_parameter/ACMix_Model_best.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                early_stop = True

print('Training completed')
if not early_stop:
    torch.save(model.state_dict(), './model_parameter/ACMix_Model.pth')
    print('Final model saved to ACMix_model.pth')
else:
    print('Best model saved to ACMix_Model_best.pth')
