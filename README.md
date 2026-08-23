# Switching Successor Measures для иерархического zero-shot RL

Репозиторий содержит реализацию **Switching Successor Measures (SSM)** для
иерархического zero-shot reinforcement learning, а также исследовательское
расширение **BiMixer**, которое за один forward pass предсказывает
упорядоченную последовательность FB-интенций.

Сайт исходного проекта - https://stestokth.github.io/switching-successors/

Отчет - cross_embodied_rl.pdf

Диск с весами, визуализации, метриками - https://drive.google.com/drive/folders/1TVL3r1o598OJ4jstndAK7x6yYi4eMFp7?usp=sharing

## Установка

Используемая конфигурация: Python 3.11.3 и GCC 12.3.0.

```bash
conda create -n ssm python=3.11
conda activate ssm
pip install -r requirements.txt
```


## Обучение BiMixer

Пример для четырёх подцелей:

```bash
python main.py \
  --env_name=ogbench-antmaze-medium-navigate-v0 \
  --agent=agents/fbpiswitch_bimixer.py \
  --agent.frozen_path=medium \
  --agent.num_subgoals=4 \
  --train_steps=500000 \
  --seed=0 \
  --wandb_run_group=bimixer_n4 \
  --enable_wandb=0 \
  --video_episodes=0
```

Параметры `F`, `B` и low-level actor загружаются из `agent.frozen_path` и
остаются замороженными. Обучается только новая high-level политика. Checkpoint
сохраняются в `exp_logs/<group>/<run>/params_<step>.pkl`.

## Оценка checkpoint

Baseline:

```bash
python evaluate_agents.py \
  --method baseline \
  --checkpoint medium/params.pkl \
  --episodes 100 \
  --seeds 0,1,2,3,4 \
  --output-dir evaluation_results/baseline
```

BiMixer:

```bash
python evaluate_agents.py \
  --method bimixer \
  --checkpoint exp_logs/bimixer_n4/sd000_20260822_121016/params_500000.pkl \
  --episodes 100 \
  --seeds 0,1,2,3,4 \
  --output-dir evaluation_results/n4
```

## Визуализация подцелей

```bash
python visualize_subgoals_pkl.py \
  --bimixer-checkpoint exp_logs/bimixer_n4/sd000_20260822_121016/params_500000.pkl \
  --frozen-path /home/makanin/rl/medium \
  --tasks all \
  --output-dir subgoal_visualizations/n4_step500k
```


## Основные результаты

Полный протокол: пять evaluation seed, 100 эпизодов на задачу и seed,
100 000 zero-shot samples. Приведены mean ± standard deviation.

| Метод | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.816 ± 0.057 | 0.838 ± 0.026 | 0.756 ± 0.021 | **0.718 ± 0.018** | 0.822 ± 0.065 | 0.790 ± 0.023 |
| BiMixer, N=3 | **0.828 ± 0.047** | 0.870 ± 0.024 | **0.808 ± 0.037** | 0.596 ± 0.073 | **0.902 ± 0.036** | **0.801 ± 0.020** |
| BiMixer, N=4 | 0.754 ± 0.036 | **0.880 ± 0.019** | 0.698 ± 0.036 | 0.566 ± 0.030 | 0.848 ± 0.026 | 0.749 ± 0.014 |
| BiMixer, N=5 | 0.754 ± 0.026 | 0.866 ± 0.015 | 0.664 ± 0.059 | 0.572 ± 0.059 | 0.764 ± 0.025 | 0.724 ± 0.023 |

`N=3` показывает лучший overall, но его преимущество над baseline невелико и
не является статистически убедительным. Увеличение `N` ухудшает результат:
при stateless-исполнении используется только `z1`, а его обучающая цель
смещается с `1/4` сегмента для `N=3` к `1/5` и `1/6` для `N=4` и `N=5`.
Подробный анализ приведён в [REPORT.md](REPORT.md).

