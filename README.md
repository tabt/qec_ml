# QEC-ML

---

###  1: Датасеты и симуляция ошибок (data/)
- **1.1** Синтетическая генерация синдромов surface code через `stim`
- **1.2** Модели шума: деполяризующий, амплитудное затухание, битовый переворот
- **1.3** Открытые датасеты: Google Sycamore syndrome данные (arXiv:2207.06431)
- **1.4** Аналоговые сигналы: симуляция IQ-данных от квантового считывания

###  2: Baseline-декодеры (decoders/)
- **2.1** MWPM через PyMatching — стандартный baseline
- **2.2** Union-Find декодер
- **2.3** Lookup Table декодер для малых кодов

###  3: ML-декодеры (models/)
- **3.1** MLP — простая нейронная сеть как первый ML baseline
- **3.2** CNN — свёрточная сеть для пространственной структуры синдромов
- **3.3** Graph Neural Network (GNN) — синдромный граф как входные данные
- **3.4** Transformer — attention по синдромным битам + временны́е серии
- **3.5** LSTM/Temporal Transformer — для коррекции аналоговых сигналов

###  4: Сравнительный анализ (benchmarks/)
- **4.1** Метрики: logical error rate, threshold, decoding time
- **4.2** Зависимость от расстояния кода d и уровня шума p
- **4.3** Scalability: сколько qubits держит каждая модель
- **4.4** Transfer learning: модель, обученная на d=5, на d=7

###  5: Коррекция аналоговых сигналов (отдельный трек)
- **5.1** IQ-классификация состояний |0⟩/|1⟩ через ML
- **5.2** Денойзинг временных серий считывания
- **5.3** Сравнение: threshold, SVM, MLP, CNN1D

### Ноутбуки
- `notebook_01_surface_code_decoding.ipynb` — основное исследование
- `notebook_02_analog_signal_correction.ipynb` — аналоговый трек

---

## Структура библиотеки `qec_ml`

```
qec_ml/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── syndrome_generator.py     # Stim-based генерация синдромов
│   ├── noise_models.py           # Модели шума
│   ├── analog_signal.py          # IQ-сигналы считывания
│   └── datasets.py               # PyTorch Dataset классы
├── decoders/
│   ├── __init__.py
│   ├── base_decoder.py           # Абстрактный класс
│   ├── mwpm_decoder.py           # MWPM (PyMatching)
│   └── lookup_decoder.py         # Lookup table
├── models/
│   ├── __init__.py
│   ├── mlp_decoder.py            # MLP
│   ├── cnn_decoder.py            # CNN
│   ├── gnn_decoder.py            # GNN (PyTorch Geometric)
│   ├── transformer_decoder.py    # Transformer
│   └── lstm_corrector.py         # LSTM для аналоговых сигналов
├── benchmarks/
│   ├── __init__.py
│   ├── metrics.py                # Метрики QEC
│   ├── runner.py                 # Benchmark runner
│   └── visualization.py          # Plotting утилиты
└── utils/
    ├── __init__.py
    ├── training.py               # Общий тренировочный цикл
    └── config.py                 # Конфиги через dataclasses
```

---

## Быстрый старт

```bash
pip install -e .
# или
pip install stim pymatching torch torch-geometric numpy matplotlib scikit-learn
```

```python
from qec_ml.data import SyndromeGenerator
from qec_ml.models import TransformerDecoder
from qec_ml.benchmarks import BenchmarkRunner

gen = SyndromeGenerator(distance=5, noise_model="depolarizing", p=0.01)
dataset = gen.generate(n_samples=10000)

model = TransformerDecoder(distance=5)
runner = BenchmarkRunner()
results = runner.compare_all(dataset)
```
