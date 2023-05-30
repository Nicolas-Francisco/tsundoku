import logging
import os
import re
import click
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import toml

from glob import glob
from multiprocessing.pool import ThreadPool
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gensim.utils import deaccent

from tsundoku.utils.files import read_toml
from tsundoku.models.transformer_pipeline import (
    execute_transformer_pipeline,
    data_loader,
)
from tsundoku.models.pipeline import search_tokens
from tsundoku.models.dataset_class import BETOTokenizer, BETOModel
from tsundoku.utils.timer import Timer


@click.command()
@click.option("--experiment", type=str, default="full")
@click.option("--group", type=str, default="location")
def main(experiment, group):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed/parquet)
    """

    experiment_name = experiment
    group_key = group

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    config = read_toml(Path(os.environ["TSUNDOKU_PROJECT_PATH"]) / "config.toml")[
        "project"
    ]
    logger.info(str(config))
    dask.config.set(pool=ThreadPool(int(config.get("n_jobs", 2))))

    source_path = Path(config["path"]["data"]) / "raw"
    experiment_file = Path(config["path"]["config"]) / "experiments.toml"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not experiment_file.exists():
        raise FileNotFoundError(experiment_file)

    with open(experiment_file) as f:
        experiment_config = toml.load(f)
        logging.info(f"{experiment_config}")

    experimental_settings = experiment_config["experiments"][experiment_name]
    logging.info(f"Experimental settings: {experimental_settings}")

    source_folders = sorted(
        glob(str(source_path / experimental_settings.get("folder_pattern", "*")))
    )
    logging.info(
        f"{len(source_folders)} folders with data. {source_folders[0]} up to {source_folders[-1]}"
    )

    key_folders = map(os.path.basename, source_folders)

    if experimental_settings.get("folder_start", None) is not None:
        key_folders = filter(
            lambda x: x >= experimental_settings.get("folder_start"), key_folders
        )

    if experimental_settings.get("folder_end", None) is not None:
        key_folders = filter(
            lambda x: x <= experimental_settings.get("folder_end"), key_folders
        )

    key_folders = list(key_folders)
    logging.info(f"{key_folders}")

    # let's go

    data_base = Path(config["path"]["data"]) / "interim"
    processed_path = (
        Path(config["path"]["data"]) / "processed" / experimental_settings.get("key")
    )

    with open(Path(config["path"]["config"]) / "groups" / f"{group_key}.toml") as f:
        group_config = toml.load(f)

    user_ids = (
        dd.read_parquet(processed_path / "user.elem_ids.parquet")
        .set_index("user.id")
        .compute()
    )
    logging.info(f"Total users: #{len(user_ids)}")

    labels = pd.DataFrame(
        0, index=user_ids.index, columns=group_config.keys(), dtype=int
    )

    user_data_path = processed_path / "user.unique.parquet"

    df = dd.read_parquet(user_data_path).set_index("user.id").compute()
    columnas_deseadas = [
        "user.description",
        "user.name",
        "user.screen_name",
        "user.url",
        "user.location",
    ]

    # Filtrar el DataFrame para mantener solo las columnas deseadas
    user_data_filtrado = df.loc[:, columnas_deseadas]
    mapeo_nombres = {
        "user.id": "id",
        "user.description": "description",
        "user.name": "name",
        "user.screen_name": "screen_name",
        "user.url": "url",
        "user.location": "location",
    }

    user_data = user_data_filtrado.rename(columns=mapeo_nombres)
    user_data["label"] = ""

    for key, meta in group_config.items():
        group_re = None
        try:
            # print(f'location patterns for {key}, {meta["location"]["patterns"]}')
            group_re = re.compile("|".join(meta["location"]["patterns"]), re.IGNORECASE)
        except KeyError:
            # print(f"no location patterns in {key}")
            continue

        user_data["location"] = user_data["location"].fillna("").map(deaccent)
        # asignamos label a todos quienes tengan un location acorde a los datos del .toml
        user_data.loc[user_data["location"].str.contains(group_re), "label"] = key
        group_ids = user_data[user_data["location"].str.contains(group_re)].index

        if group == "location":
            # use these as account ids that cannot be modified (let's trust users)
            if not "account_ids" in meta:
                meta["account_ids"] = dict()

            if not "known_users" in meta:
                meta["account_ids"]["known_users"] = list(group_ids)
            else:
                meta["account_ids"]["known_users"].extend(group_ids)
        else:
            # use them as labels
            labels[key].loc[group_ids] = 1

    # xgb_parameters = experiment_config["location"]["xgb"]
    # pipeline_config = experiment_config["location"]["pipeline"]

    df = user_data

    # print(f"df shape: {df.shape}")

    # training_df = df[df["location"] != ""]

    # print(f"training_df shape: {training_df.shape}")

    # training_df = df[df["label"] != ""]

    # print(f"training_df_labeled shape: {training_df.shape}")

    # print(training_df.head(5))

    # unique_values = df["label"].unique()

    # print(unique_values)

    MAX_LEN = 200
    BATCH_SIZE = 20

    training_df = df[df["label"] != ""]

    df_train, df_validation, df_test = np.split(
        training_df.sample(frac=1),
        [int(0.7 * len(training_df)), int(0.8 * len(training_df))],
    )

    train_data_loader = data_loader(df_train, BETOTokenizer, MAX_LEN, BATCH_SIZE)
    validation_data_loader = data_loader(
        df_validation, BETOTokenizer, MAX_LEN, BATCH_SIZE
    )
    test_data_loader = data_loader(df_test, BETOTokenizer, MAX_LEN, BATCH_SIZE)

    t = Timer()
    chronometer = []
    t.start()

    execute_transformer_pipeline(
        train_data_loader, df_train, validation_data_loader, df_validation
    )

    return

    # clf, predictions, feature_names_all, top_terms, X = classifier_pipeline(
    #     processed_path,
    #     group_config,
    #     user_ids,
    #     labels,
    #     xgb_parameters,
    #     allowed_user_ids=allow_list_ids,
    #     allowed_users_class=allow_id_class,
    #     early_stopping_rounds=pipeline_config["early_stopping_rounds"],
    #     eval_fraction=pipeline_config["eval_fraction"],
    #     threshold_offset_factor=pipeline_config["threshold_offset_factor"],
    #     skip_numeric_tokens=skip_numeric_tokens,
    # )

    current_timer = t.stop()

    logger.info("Chronometer: " + str(chronometer))
    logger.info("Chronometer process name: location clasification with transformers")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
