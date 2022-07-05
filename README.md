# NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning: 

NodeGAM is an interpretable deep learning GAM model proposed in our ICLR 2022 paper: 
[NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/abs/2106.01613).
In short, it trains a GAM model by multi-layer differentiable trees to be accurate, interpretable, and 
differentiable. See this [blog post](https://medium.com/@chkchang21/interpretable-deep-learning-models-for-tabular-data-neural-gams-500c6ecc0122) 
for an intro, and our [documentation website](https://nodegam.readthedocs.io/en/latest/)!


<img src="https://github.com/zzzace2000/nodegam/blob/main/resources/images/Fig1.png?raw=true" width=600px>

## Installation

```bash
pip install nodegam
```

## The performance and the runtime of the NodeGAM package

We compare NodeGAM with other GAMs (EBM, XGB-GAM), and XGB in 6 datasets.
All models use default parameters, so the performance of NodeGAM here is lower than what paper 
reported. We find NodeGAM often performs better in larger datasets.

3 classification datasets:

| Dataset/AUROC | Domain   | N    | P  | NodeGAM           | EBM               | XGB-GAM       | XGB               |
|---------------|----------|------|----|-------------------|-------------------|---------------|-------------------|
| MIMIC-II      | Medicine | 25K  | 17 | 0.844 ± 0.018     | 0.842 ± 0.019     | 0.833 ± 0.02  | **0.845 ± 0.019** |
| Adult         | Finance  | 33K  | 14 | 0.916 ± 0.002     | **0.927 ± 0.003** | 0.925 ± 0.002 | **0.927 ± 0.002** |
| Credit        | Finance  | 285K | 30 | **0.989 ± 0.008** | 0.984 ± 0.007     | 0.985 ± 0.008 | 0.984 ± 0.01      |

3 regression datasets:

| Dataset/RMSE | Domain | N    | P  | NodeGAM           | EBM            | XGB-GAM         | XGB                |
|--------------|--------|------|----|-------------------|----------------|-----------------|--------------------|
| Wine         | Nature | 5K   | 12 | 0.705 ± 0.012     | 0.69 ± 0.011   | 0.713 ± 0.006   | **0.682 ± 0.023**  |
| Bikeshare    | Retail | 17K  | 16 | 57.438 ± 3.899    | 55.676 ± 0.327 | 101.093 ± 0.946 | **45.212 ± 1.254** |
| Year         | Music  | 515K | 90 | **9.013 ± 0.004** | 9.204 ± 0.0    | 9.257 ± 0.0     | 9.049 ± 0.0        |

We also find the run time of our model increases mildly with growing data size due to mini-batch 
training, while our baselines increase training time much more.

3 classification datasets:

| Dataset/Seconds | Domain   | N    | P  | NodeGAM      | EBM        | XGB-GAM       | XGB            |
|-----------------|----------|------|----|--------------|------------|---------------|----------------|
| MIMIC-II        | Medicine | 25K  | 17 | 105.0 ± 14.0 | 6.0 ± 2.0  | **0.0 ± 1.0** | 1.0 ± 1.0      |
| Adult           | Finance  | 33K  | 14 | 196.0 ± 56.0 | 15.0 ± 8.0 | 6.0 ± 0.0     | **1.0 ± 0.0**  |
| Credit          | Finance  | 285K | 30 | 113.0 ± 36.0 | 37.0 ± 2.0 | 26.0 ± 7.0    | **16.0 ± 2.0** |

3 regression datasets:

| Dataset/Seconds | Domain | N    | P  | NodeGAM          | EBM         | XGB-GAM       | XGB           |
|-----------------|--------|------|----|------------------|-------------|---------------|---------------|
| Wine            | Nature | 5K   | 12 | 157.0 ± 86.0     | 4.0 ± 2.0   | **0.0 ± 0.0** | **0.0 ± 0.0** |
| Bikeshare       | Retail | 17K  | 16 | 223.0 ± 23.0     | 15.0 ± 3.0  | **1.0 ± 1.0** | 2.0 ± 1.0     |
| Year            | Music  | 515K | 90 | **318.0 ± 20.0** | 501.0 ± 8.0 | 376.0 ± 1.0   | 537.0 ± 1.0   |

Reproducing notebook is [here](https://nbviewer.jupyter.org/github/zzzace2000/nodegam/blob/main/notebooks/benchmark%20speed%20and%20acc%20of%20the%20package.ipynb).

See the Table 1 and 2 of our [paper](https://arxiv.org/abs/2106.01613) for more comparisons.


## NodeGAM Training

### Sklearn interface

To simply use it on your dataset, just run:
```python
from nodegam.sklearn import NodeGAMClassifier, NodeGAMRegressor

model = NodeGAMClassifier()
model.fit(X, y)
```

Understand the model:
```python
model.visualize()
```

or

```python
from nodegam.vis_utils import vis_GAM_effects

vis_GAM_effects({
    'nodegam': model.get_GAM_df(),
})
```

<img src="https://github.com/zzzace2000/nodegam/blob/main/resources/images/example_toy_nodegam.png?raw=true" width=600px>

See the `notebooks/toy dataset with nodegam sklearn.ipynb` [here](https://nbviewer.jupyter.org/github/zzzace2000/nodegam/blob/main/notebooks/toy%20dataset%20with%20nodegam%20sklearn.ipynb).


### Notebook training

It is useful if you want to customize the NodeGAM training to your PyTorch pipeline. 
You can find details of the training in this notebook:
https://colab.research.google.com/drive/1C_gBoSc1AlQ7VvCXVWiU-7X3YjQZTiZI?usp=sharing

And see more examples under `notebooks/`

### Python file

You can also train a NodeGAM using our main file.
To reproduce our results, e.g. NODE-GA2M trained in fold 0 (total 5 folds) of bikeshare, you can run
```bash
hparams="resources/best_hparams/node_ga2m/0519_f0_best_bikeshare_GAM_ga2m_s83_nl4_nt125_td1_d6_od0.0_ld0.3_cs0.5_lr0.01_lo0_la0.0_pt0_pr0_mn0_ol0_ll1"
python main.py \ 
--name 0603_best_bikeshare_f0 \ 
--load_from_hparams ${hparams}
--fold 0
```
The models will be stored in `logs/0603_best_bikeshare_f0/`. And the results including test/val error are stored in `results/bikeshare_GAM.csv`

We provide the best hyperparmeters we found in `best_hparams/`.

## Baseline GAMs

We also provide code to train other GAMs for comparisons such as:
- Spline: we use the [pygam](https://pygam.readthedocs.io/en/latest/) package.
- EBM: [Explainable Boosting Machine](https://github.com/interpretml/interpret).
- XGB-GAM: Limit the XGB to have tree depth 1 that removes all interaction effects in the model. 
It's proposed in [our KDD paper](https://arxiv.org/abs/2006.06466).  

### Sklearn interface

To train baselines on your dataset, just run:

```python
from nodegam.gams.MySpline import MySplineLogisticGAM, MySplineGAM
from nodegam.gams.MyEBM import MyExplainableBoostingClassifier, MyExplainableBoostingRegressor
from nodegam.gams.MyXGB import MyXGBOnehotClassifier, MyXGBOnehotRegressor
from nodegam.gams.MyBagging import MyBaggingClassifier, MyBaggingRegressor


ebm = MyExplainableBoostingClassifier()
ebm.fit(X, y)

spline = MySplineLogisticGAM()
bagged_spline = MyBaggingClassifier(base_estimator=spline, n_estimators=3)
bagged_spline.fit(X, y)

xgb_gam = MyXGBOnehotClassifier()
bagged_xgb = MyBaggingClassifier(base_estimator=xgb_gam, n_estimators=3)
bagged_xgb.fit(X, y)
```

Understand the models:

```python
from nodegam.vis_utils import vis_GAM_effects

fig, ax = vis_GAM_effects(
    all_dfs={
        'EBM': ebm.get_GAM_df(),
        'Spline': bagged_spline.get_GAM_df(),
        'XGB-GAM': bagged_xgb.get_GAM_df(),
    },
)
```

<img src="https://github.com/zzzace2000/nodegam/blob/main/resources/images/example_gam_plot.png?raw=true" width=600px>

See the `notebooks/toy dataset with nodegam sklearn.ipynb` [here](https://nbviewer.jupyter.org/github/zzzace2000/nodegam/blob/main/notebooks/toy%20dataset%20with%20nodegam%20sklearn.ipynb) for an example.


### Python file

You can train Spline, EBM, and XGB-GAM by the following commands.

```bash
python baselines.py --name 0603_bikeshare_spline_f0 --fold 0 --model_name spline --dataset bikeshare
python baselines.py --name 0603_bikeshare_ebm_f0 --fold 0 --model_name ebm-o100-i100 --dataset bikeshare
python baselines.py --name 0603_bikeshare_xgb-o5_f0 --fold 0 --model_name xgb-o5 --dataset bikeshare
```

The result is shown in `results/baselines_bikeshare.csv` and the model is stored in `logs/{name}/`.

## Visualization of the trained models stored under `logs/`

To visualize and compare multiple trained GAM models stored under `logs/`, run this in a notebook:

```python
from nodegam.vis_utils import vis_GAM_effects
from nodegam.utils import average_GAMs

df_dict = {
    'node_ga2m': average_GAMs([
        '0603_best_bikeshare_f0',
        '0603_best_bikeshare_f1',
    ], max_n_bins=256),
    'ebm': average_GAMs([
        '0603_bikeshare_ebm_f0',
        '0603_bikeshare_ebm_f1',
    ], max_n_bins=256),
}

fig, ax = vis_GAM_effects(df_dict)
```

<img src="https://github.com/zzzace2000/nodegam/blob/main/resources/images/example_bikeshare_main.png?raw=true" width=600px>

To avoid long computations, when visualizing we specify max_n_bins to do quantile binning of each 
feature to have at most 256 bins (default). The `average_GAMs` take average of multiple runs of GAMs to produce mean and stdev on the GAM 
graphs.

See `notebooks/bikeshare visualization.ipynb` [here](https://nbviewer.org/github/zzzace2000/nodegam/blob/main/notebooks/bikeshare%20visualization.ipynb) which we show bikeshare graphs for all GAMs 
(NODE-GA2M, NODE-GAM, EBM and Spline) in our paper.


## Citations

If you find the code useful, please cite:
```
@inproceedings{chang2021node,
  title={NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning},
  author={Chang, Chun-Hao and Caruana, Rich and Goldenberg, Anna},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{chang2021interpretable,
  title={How interpretable and trustworthy are gams?},
  author={Chang, Chun-Hao and Tan, Sarah and Lengerich, Ben and Goldenberg, Anna and Caruana, Rich},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={95--105},
  year={2021}
}
```


## Contributing

All content in this repository is licensed under the MIT license.
