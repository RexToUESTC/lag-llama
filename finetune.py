from importlib.metadata import metadata
from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from torch.distributed.checkpoint import Metadata
from tqdm.autonotebook import tqdm

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
import pandas as pd

from lag_llama.gluon.estimator import LagLlamaEstimator

# dataset = get_dataset("m4_weekly")
#
# prediction_length = dataset.metadata.prediction_length
# print(prediction_length)
# context_length = prediction_length*3

from gluonts.dataset.common import MetaData as mD

import pandas as pd
from gluonts.dataset.common import TrainDatasets

import numpy as np


def advanced_csv_to_gluonts_dataset(
        csv_path: str,
        timestamp_column: str,
        target_column: str,
        item_id_column: str = None,
        static_cat_columns: list = None,
        static_real_columns: list = None,
        dynamic_real_columns: list = None,
        freq: str = "D",
        prediction_length: int = 7
) -> TrainDatasets:
    df = pd.read_csv(csv_path)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], utc=True)
    # 确保目标列是float32类型
    df[target_column] = df[target_column].astype(np.float32)

    df = df.set_index(timestamp_column)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df

    dataset_train = PandasDataset(
            train_df,
            target=target_column,
            freq=freq
        )

    dataset_test = PandasDataset(
            test_df,
            target=target_column,
            freq=freq
        )

    metadata=mD(freq=freq,prediction_length=prediction_length)

    return TrainDatasets(
        metadata=metadata,
        train=dataset_train,
        test=dataset_test
    )

dataset = advanced_csv_to_gluonts_dataset(
    csv_path="energy_dataset_dealed.csv",
    timestamp_column="time",
    target_column="price actual",
    freq="1H",
    prediction_length=24
)
prediction_length = dataset.metadata.prediction_length
print(prediction_length)
context_length = prediction_length * 3

num_samples = 20
device = "cuda"

ckpt = torch.load("lag-llama.ckpt", map_location=device)
estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

estimator = LagLlamaEstimator(
    ckpt_path="lag-llama.ckpt",
    prediction_length=prediction_length,
    context_length=context_length,

    # distr_output="neg_bin",
    # scaling="mean",
    nonnegative_pred_samples=True,
    aug_prob=0,
    lr=5e-4,

    # estimator args
    input_size=estimator_args["input_size"],
    n_layer=estimator_args["n_layer"],
    n_embd_per_head=estimator_args["n_embd_per_head"],
    n_head=estimator_args["n_head"],
    time_feat=estimator_args["time_feat"],

    # rope_scaling={
    #     "type": "linear",
    #     "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    # },

    batch_size=64,
    num_parallel_samples=num_samples,
    trainer_kwargs={"max_epochs": 50, },  # <- lightning trainer arguments
)

predictor = estimator.train(dataset.train, cache_data=True, shuffle_buffer_length=1000)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
    num_samples=num_samples
)

forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))

tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

plt.figure(figsize=(20, 15))
date_formater = mdates.DateFormatter('%b, %d')
plt.rcParams.update({'font.size': 15})

# Iterate through the first 9 series, and plot the predicted samples
for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
    ax = plt.subplot(3, 3, idx + 1)

    plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target", )
    forecast.plot(color='g')
    plt.xticks(rotation=60)
    ax.xaxis.set_major_formatter(date_formater)
    ax.set_title(forecast.item_id)

plt.gcf().tight_layout()
plt.legend()
plt.show()

evaluator = Evaluator()
agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))

print(agg_metrics)
print(ts_metrics)
