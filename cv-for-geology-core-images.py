"""Классификация горных пород по изображениям керна

Модель компьютерного зрения для автоматической классификации типов горных пород
по изображениям керна с использованием сверточных нейронных сетей.
"""

# =============================================================================
# ИМПОРТ БИБЛИОТЕК И СОЗДАНИЕ ПАПОК
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import joblib
from pathlib import Path

# Создаем структуру папок
print("📁 Создание структуры папок...")
folders = ['images', 'results', 'models', 'data']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✅ Создана папка: {folder}/")

# =============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# =============================================================================
print("🔄 Загрузка датасета...")

# Установка kagglehub для загрузки данных
import subprocess
import sys
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_package('kagglehub')

import kagglehub

# Загрузка датасета
path = kagglehub.dataset_download("stealthtechnologies/rock-classification")
print(f'📦 Датасет загружен в: {path}')

# Анализ структуры данных
print("🔍 Анализ структуры данных...")

def analyze_dataset(path):
    """Анализ структуры датасета"""
    total_images = len(list(Path(path).rglob('*.jpg'))) + len(list(Path(path).rglob('*.png')))
    folders = {}

    for p in Path(path).rglob('*.jpg'):
        folder_name = p.parent.name
        folders[folder_name] = folders.get(folder_name, 0) + 1

    print(f"📊 Общее количество изображений: {total_images}")
    print(f"📁 Структура папок: {folders}")

    return total_images, folders

total_images, folder_structure = analyze_dataset(path)

# Сбор метаданных
print("\n📥 Сбор метаданных изображений...")

metadata = []

def collect_images_from_folder(folder_path, data_type):
    images = []
    if os.path.exists(folder_path):
        for class_name in os.listdir(folder_path):
            class_path = os.path.join(folder_path, class_name)

            if os.path.isdir(class_path):
                class_images = []
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        full_path = os.path.join(class_path, file)
                        class_images.append(full_path)

                        metadata.append({
                            'file_path': full_path,
                            'class_name': class_name,
                            'data_type': data_type
                        })

                print(f"   🎯 {class_name}: {len(class_images)} изображений")
                images.extend(class_images)
    return images

rock_data_path = os.path.join(path, 'Rock Data')
data_types = ['train', 'test', 'valid']

train_images, test_images, valid_images = [], [], []

for data_type in data_types:
    folder = os.path.join(rock_data_path, data_type)
    if data_type == 'train':
        train_images = collect_images_from_folder(folder, data_type)
    elif data_type == 'test':
        test_images = collect_images_from_folder(folder, data_type)
    elif data_type == 'valid':
        valid_images = collect_images_from_folder(folder, data_type)

print(f"\n✅ Итоговые результаты:")
print(f"   🏋️  Обучающая выборка: {len(train_images)} изображений")
print(f"   🧪 Тестовая выборка: {len(test_images)} изображений")
print(f"   📊 Валидационная выборка: {len(valid_images)} изображений")
print(f"   📈 Всего: {len(metadata)} записей в метаданных")

# =============================================================================
# СОЗДАНИЕ DATASET И DATALOADER
# =============================================================================
class RockDataset(Dataset):
    """Кастомный Dataset для загрузки изображений горных пород"""

    def __init__(self, metadata_list, transform=None):
        self.metadata = metadata_list
        self.transform = transform

        self.classes = sorted(list(set([item['class_name'] for item in metadata_list])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        file_path = item['file_path']
        class_name = item['class_name']

        image = Image.open(file_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[class_name]
        return image, label

# Разделение метаданных
train_metadata = [item for item in metadata if item['data_type'] == 'train']
test_metadata = [item for item in metadata if item['data_type'] == 'test']
valid_metadata = [item for item in metadata if item['data_type'] == 'valid']

# Аугментация данных
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создание datasets и dataloaders
train_dataset = RockDataset(train_metadata, transform=train_transform)
valid_dataset = RockDataset(valid_metadata, transform=basic_transform)
test_dataset = RockDataset(test_metadata, transform=basic_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"\n📦 Datasets созданы:")
print(f"   🎯 Количество классов: {len(train_dataset.classes)}")
print(f"   📝 Классы: {train_dataset.classes}")

# =============================================================================
# СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 Используемое устройство: {device}")

def create_rock_classifier(num_classes):
    """Создание модели для классификации горных пород"""
    model = models.resnet18(pretrained=True)

    # Заморозка весов
    for param in model.parameters():
        param.requires_grad = False

    # Замена последнего слоя
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Разморозка последнего слоя
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

# Создание модели
model = create_rock_classifier(num_classes=len(train_dataset.classes))
model = model.to(device)

print("✅ Модель создана!")
print(f"🎯 Количество классов: {len(train_dataset.classes)}")

# Настройка обучения
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs=10):
    """Функция обучения модели"""
    train_losses = []
    valid_accuracies = []
    best_accuracy = 0.0

    print("🚀 Начало обучения...")

    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Метрики
        train_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        train_losses.append(train_loss)
        valid_accuracies.append(accuracy)

        scheduler.step()

        # Сохранение лучшей модели
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'models/best_rock_model.pth')

        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {train_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, '
              f'Best: {best_accuracy:.2f}%')

    return train_losses, valid_accuracies

# Обучение базовой модели
print("🎯 Обучение базовой модели...")
train_losses, valid_accuracies = train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=10
)

# =============================================================================
# ТЕСТИРОВАНИЕ И ОЦЕНКА
# =============================================================================
print("🧪 Тестирование модели...")

# Загрузка лучшей модели
model.load_state_dict(torch.load('models/best_rock_model.pth'))
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"✅ Точность на тестовой выборке: {test_accuracy:.2f}%")

# =============================================================================
# УЛУЧШЕННАЯ МОДЕЛЬ С FINE-TUNING
# =============================================================================
print("\n🔧 Создание улучшенной модели с fine-tuning...")

def create_improved_model(num_classes):
    """Создание улучшенной модели с размороженными слоями"""
    model = models.resnet18(pretrained=True)

    # Заморозка всех слоев
    for param in model.parameters():
        param.requires_grad = False

    # Разморозка последних слоев
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True

    # Улучшенный классификатор
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model

# Создание и обучение улучшенной модели
model_improved = create_improved_model(len(train_dataset.classes))
model_improved = model_improved.to(device)

# Оптимизатор с разными learning rates
optimizer_improved = torch.optim.Adam([
    {'params': model_improved.layer3.parameters(), 'lr': 0.0001},
    {'params': model_improved.layer4.parameters(), 'lr': 0.0001},
    {'params': model_improved.fc.parameters(), 'lr': 0.001}
])

print("🚀 Обучение улучшенной модели...")
train_losses_improved, valid_accuracies_improved = train_model(
    model=model_improved,
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer_improved,
    scheduler=scheduler,
    epochs=15
)

# Тестирование улучшенной модели
model_improved.load_state_dict(torch.load('models/best_rock_model.pth'))
model_improved.eval()

test_correct_improved = 0
test_total_improved = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_improved(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total_improved += labels.size(0)
        test_correct_improved += (predicted == labels).sum().item()

test_accuracy_improved = 100 * test_correct_improved / test_total_improved
print(f"✅ Точность улучшенной модели: {test_accuracy_improved:.2f}%")

# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n📊 Визуализация результатов...")

plt.figure(figsize=(15, 5))

# График потерь
plt.subplot(1, 3, 1)
plt.plot(train_losses, 'b-', label='Базовая модель', linewidth=2)
plt.plot(train_losses_improved, 'r-', label='Улучшенная модель', linewidth=2)
plt.title('Функция потерь во время обучения')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)

# График точности
plt.subplot(1, 3, 2)
plt.plot(valid_accuracies, 'b-', label='Базовая модель', linewidth=2)
plt.plot(valid_accuracies_improved, 'r-', label='Улучшенная модель', linewidth=2)
plt.title('Точность на валидационной выборке')
plt.xlabel('Эпоха')
plt.ylabel('Точность (%)')
plt.legend()
plt.grid(True)

# Сравнение моделей
plt.subplot(1, 3, 3)
models_names = ['Базовая\nмодель', 'Улучшенная\nмодель']
accuracies = [test_accuracy, test_accuracy_improved]
colors = ['blue', 'red']

bars = plt.bar(models_names, accuracies, color=colors, alpha=0.7)
plt.title('Сравнение точности моделей')
plt.ylabel('Точность на тесте (%)')
plt.ylim(0, 100)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/training_results.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================
print("\n💾 Сохранение результатов...")

# Сохранение метрик
results = {
    'basic_model_accuracy': test_accuracy,
    'improved_model_accuracy': test_accuracy_improved,
    'improvement': test_accuracy_improved - test_accuracy,
    'num_classes': len(train_dataset.classes),
    'class_names': train_dataset.classes,
    'dataset_stats': {
        'train_images': len(train_images),
        'test_images': len(test_images),
        'valid_images': len(valid_images),
        'total_images': total_images
    }
}

with open('results/training_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Сохранение информации о модели
model_info = {
    'best_model_path': 'models/best_rock_model.pth',
    'input_size': 224,
    'num_classes': len(train_dataset.classes),
    'class_mapping': train_dataset.class_to_idx,
    'transform_info': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}

with open('results/model_info.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print("✅ Результаты сохранены:")
print("   📊 results/training_results.json - метрики обучения")
print("   🔧 results/model_info.json - информация о модели")
print("   🖼️  images/training_results.png - графики обучения")

# =============================================================================
# ФИНАЛЬНЫЙ ОТЧЕТ
# =============================================================================
print("\n" + "="*60)
print("🎉 ФИНАЛЬНЫЙ ОТЧЕТ ПРОЕКТА")
print("="*60)

print(f"🏆 Лучшая модель: Улучшенная ResNet18")
print(f"📊 Метрики:")
print(f"   • Точность базовой модели: {test_accuracy:.2f}%")
print(f"   • Точность улучшенной модели: {test_accuracy_improved:.2f}%")
print(f"   • Улучшение: {test_accuracy_improved - test_accuracy:+.2f}%")

print(f"\n📁 Данные:")
print(f"   • Количество классов: {len(train_dataset.classes)}")
print(f"   • Обучающая выборка: {len(train_images)} изображений")
print(f"   • Тестовая выборка: {len(test_images)} изображений")
print(f"   • Всего изображений: {total_images}")

print(f"\n🔧 Особенности реализации:")
print(f"   • Transfer learning с ResNet18")
print(f"   • Fine-tuning последних слоев")
print(f"   • Аугментация данных для улучшения обобщения")
print(f"   • Регуляризация для борьбы с переобучением")

print(f"\n📁 СТРУКТУРА ПРОЕКТА:")
print("   rock-classification-cv/")
print("   ├── rock_classification.py     # Основной скрипт")
print("   ├── requirements.txt          # Зависимости")
print("   ├── README.md                 # Документация")
print("   ├── models/                   # Сохраненные модели")
print("   ├── results/                  # Метрики и результаты")
print("   ├── images/                   # Графики и визуализации")
print("   └── data/                     # Данные (автозагрузка)")

print("\n✅ ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
