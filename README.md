# NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning: 

NodeGAM is an interpretable deep learning GAM model proposed in our ICLR 2022 paper: 
[NODE GAM: Differentiable Generalized Additive Model for Interpretable Deep Learning](https://arxiv.org/abs/2106.01613).
In short, it trains a GAM model by multi-layer differentiable trees to be accurate, interpretable, and 
differentiable. See this [blog post](https://medium.com/@chkchang21/interpretable-deep-learning-models-for-tabular-data-neural-gams-500c6ecc0122) for details.


<img src="https://github.com/zzzace2000/nodegam/blob/main/resources/images/Fig1.png?raw=true" width=600px>

## Installation

```bash
pip install nodegam
```

## Introducing the NodeGAM

NodeGAM compared to other GAMs (EBM, XGB-GAM), and XGB.
We find NodeGAM often performs better in larger datasets.

3 classification datasets:

| Dataset/AUROC | Domain   | N    | P  | NodeGAM           | EBM               | XGB-GAM       | XGB               |
|---------------|----------|------|----|-------------------|-------------------|---------------|-------------------|
| MIMIC-II      | Medicine | 25K  | 17 | **0.843 ± 0.018** | 0.842 ± 0.019     | 0.833 ± 0.019 | **0.843 ± 0.02**  |
| Adult         | Finance  | 33K  | 14 | 0.916 ± 0.003     | **0.927 ± 0.003** | 0.925 ± 0.002 | **0.927 ± 0.002** |
| Credit        | Finance  | 285K | 30 | **0.989 ± 0.005** | 0.984 ± 0.007     | 0.986 ± 0.008 | 0.983 ± 0.009     |

3 regression datasets:

| Dataset/RMSE | Domain | N    | P  | NodeGAM        | EBM            | XGB-GAM         | XGB                |
|--------------|--------|------|----|----------------|----------------|-----------------|--------------------|
| Wine         | Nature | 5K   | 12 | 0.703 ± 0.011  | 0.69 ± 0.011   | 0.714 ± 0.007   | **0.688 ± 0.013**  |
| Bikeshare    | Retail | 17K  | 16 | 54.711 ± 0.736 | 55.676 ± 0.327 | 101.072 ± 0.995 | **45.364 ± 1.155** |
| Year         | Music  | 515K | 90 | **9.031**      | 9.204          | 9.255           | 9.066              |

We find the run time of our model increases mildly with growing data size due to mini-batch 
training, while our baselines increase training time in several magnitudes.

3 classification datasets:

| Dataset/Seconds | Domain   | N    | P  | NodeGAM  | EBM       | XGB-GAM       | XGB           |
|-----------------|----------|------|----|----------|-----------|---------------|---------------|
| MIMIC-II        | Medicine | 25K  | 17 | 94 ± 16  | 5.0 ± 1.0 | **0.0 ± 0.0** | 0.7 ± 0.6     |
| Adult           | Finance  | 33K  | 14 | 113 ± 25 | 11.0 ± 1  | 6 ± 3         | **1.7 ± 0.6** |
| Credit          | Finance  | 285K | 30 | 122 ± 27 | 36 ± 1    | 21 ± 6        | **14 ± 1**    |

3 regression datasets:

| Dataset/Seconds | Domain | N    | P  | NodeGAM  | EBM       | XGB-GAM       | XGB           |
|-----------------|--------|------|----|----------|-----------|---------------|---------------|
| Wine            | Nature | 5K   | 12 | 104 ± 24 | 3.3 ± 1.5 | **0.0 ± 0.0** | **0.0 ± 0.0** |
| Bikeshare       | Retail | 17K  | 16 | 222 ± 31 | 15 ± 2    | **1.0 ± 0.0** | 1.7 ± 0.6     |
| Year            | Music  | 515K | 90 | **266**  | 496       | 310           | 399           |


To see the full result, see the Table 1 and 2 of our [paper](https://arxiv.org/abs/2106.01613).


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
