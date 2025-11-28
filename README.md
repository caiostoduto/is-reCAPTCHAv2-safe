# is-reCAPTCHAv2-safe

Classificação de imagens de desafios do Google reCAPTCHA v2 para estudo de segurança (ex.: "Traffic Light", "Bus", etc.). O foco é criar um pipeline reproduzível: coleta de múltiplas fontes públicas → unificação → conversão pra HDFa5 → treinamento e avaliação de modelos simples (CNN) e testes com arquiteturas mais avançadas (timm / YOLO).

> Aviso: "reCAPTCHA" é marca registrada do Google. Este repositório usa apenas dados publicamente distribuídos já disponíveis em outros locais para fins de pesquisa. Verifique licenças/termos antes de redistribuir.

## Sumário

1. Estrutura do Repositório
2. Instalação
3. Treinamento (CNN Simples)
4. Modelos Avançados (timm / YOLO)

## 1. Estrutura do Repositório

```
pyproject.toml        Configuração e dependências (Python >= 3.13)
datasets/             Arquivos baixados + parquet + HDF5 (gerados)
src/
	utils/
		dataset_download.py  Download dos datasets
		load_datasets.py     Consolidação, normalização de rótulos, deduplicação
	pytorch.py             Pipeline de treino/eval + saída de métricas
	timm.ipynb             Notebook para arquiteturas timm
	yolo.ipynb             Notebook para testes com YOLO (ultralytics)
```

## 2. Instalação

Requer Python 3.13.

Requer CUDA e Pytorch devidamente instalados: https://pytorch.org/get-started/locally/

```bash
pip install -e .
```

ou

```bash
uv sync
```

## 3. Treinamento (CNN Simples)

Script: `src/pytorch.py`.

Rodar:

```bash
python src/pytorch.py
```

Saídas em `./is_recaptchav2_safe/pytorch/`:

- `results.csv` (loss por epoch)
- `results.txt` (accuracy final)
- `confusion_matrix.png` e `confusion_matrix_normalized.png`

Transformações aplicadas no loader: `RandomHorizontalFlip` + `Normalize`. Não há `ToTensor` porque imagens já estão em tensor dentro do HDF5.

Seleção de dispositivo automática: CUDA → MPS → CPU.

## 4. Experimentos Avançados

### timm

Usar `timm.ipynb` para testar arquiteturas (ex. `resnet50`, `efficientnet_v2_s`).

### YOLO (ultralytics)

Notebook `yolo.ipynb` pode explorar se as classes mapeadas servem para fine-tuning de detecção (exige anotação bounding box que não existe hoje — requer etapa adicional se for perseguido).
