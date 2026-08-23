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

