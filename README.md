# Is reCAPTCHAv2 Safe?

Este projeto investiga a segurança do reCAPTCHAv2 através do treinamento de modelos de classificação de imagens (CNN e YOLO) para resolver desafios de imagem do reCAPTCHA.

## Requisitos

- Python 3.13+
- GPU com suporte CUDA (opcional, mas recomendado)
- Pelo menos 8GB de RAM

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/caiostoduto/is-reCAPTCHAv2-safe.git
cd is-reCAPTCHAv2-safe
```

2. Instale as dependências:
```bash
uv sync
```

As dependências incluem:
- PyTorch e Torchvision
- Ultralytics (YOLO)
- H5py para armazenamento eficiente de dados
- Scikit-learn para métricas
- Pandas e NumPy para manipulação de dados

## 📁 Estrutura de Dados

O projeto utiliza uma estrutura de dados com validação cruzada (k-fold):

```
dataset_fold{0-4}/
├── labels.txt          # Arquivo com labels e splits
├── train.h5           # Dataset de treino em HDF5 (CNN)
├── val.h5             # Dataset de validação em HDF5 (CNN)
├── train/             # Diretório de treino (YOLO)
│   ├── Bicycle/
│   ├── Bridge/
│   ├── Bus/
│   └── ...
└── val/               # Diretório de validação (YOLO)
    ├── Bicycle/
    ├── Bridge/
    └── ...
```

## Modelo 1: CNN (PyTorch)

### Arquiteturas Disponíveis

O código oferece três arquiteturas CNN:

1. **SimpleCNN**: Modelo básico com 2 blocos convolucionais
2. **BetterCNN**: Modelo melhorado com 3 blocos convolucionais
3. **BetterImprovedCNN**: Modelo avançado com BatchNorm, Dropout e AdaptiveAvgPool

### Como Executar

```bash
cd src
python pytorch.py
```

## Modelo 2: YOLO (Ultralytics)

### Modelos Disponíveis

O projeto suporta os seguintes modelos de classificação YOLO:

**YOLOv8**:
- `yolov8n-cls.pt` (Nano)
- `yolov8s-cls.pt` (Small)
- `yolov8m-cls.pt` (Medium)
- `yolov8l-cls.pt` (Large)
- `yolov8x-cls.pt` (Extra Large)

**YOLO11**:
- `yolo11n-cls.pt` (Nano)
- `yolo11s-cls.pt` (Small)
- `yolo11m-cls.pt` (Medium)
- `yolo11l-cls.pt` (Large)
- `yolo11x-cls.pt` (Extra Large)

### Como Executar

1. Abra o Jupyter Notebook
2. Execute as células em ordem