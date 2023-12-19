# Copyright 2023 The US Pilot Synthetic Data Gen Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to generate impressions for a single Campaign."""

from dataclasses import dataclass, asdict
from generate import generate
import random
import datetime
import csv
import pandas as pd
import time
import argparse
import numpy as np
import math
import multiprocessing

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--randomseed", help="Random seed", type=int)
parser.add_argument("-e", "--edpname", help="Edp name")


def getPlatformDistSpec(configRow):
    desktop = configRow["Platform_Desktop"]
    mobile = configRow["Platform_Mobile"]

    assert desktop + mobile == 1

    return [
        ("desktop", desktop),
        ("mobile", mobile),
    ]


def getGenderDistSpec(configRow):
    male = configRow["Gender(Male)"]
    female = configRow["Gender(Female)"]

    assert male + female == 1

    return [
        ("male", male),
        ("female", female),
    ]


def getAgeDistSpec(configRow):
    age_18_24 = configRow["Age(18-24)"]
    age_25_34 = configRow["Age(25-34)"]
    age_35_44 = configRow["Age(35-44)"]
    age_45_54 = configRow["Age(45-54)"]
    age_55_64 = configRow["Age(55-64)"]
    age_65_plus = configRow["Age(65+)"]

    assert age_18_24 + age_25_34 + age_35_44 + age_45_54 + age_55_64 + age_65_plus == 1

    return [
        ("age_18_24", age_18_24),
        ("age_25_34", age_25_34),
        ("age_35_44", age_35_44),
        ("age_45_54", age_45_54),
        ("age_55_64", age_55_64),
        ("age_65_plus", age_65_plus),
    ]


def getRealFreqDistSpec(configRow):
    mappingDict = {1: "Frequency1", 2: "Frequency2", 3: "Frequency3", 4: "Frequency4", 5: "Frequency5+"}
    abs_total_reach = configRow["Total Reach"] * configRow["Impressions"]
    mappedResult = [(key, configRow[mappingDict[key]] * abs_total_reach) for key in mappingDict.keys()]
    assert abs_total_reach == sum([val[1] for val in mappedResult])
    return mappedResult


def generate_and_analyze_for_edp(key, configRow, randomSeed):
    print(f"START {configRow}")
    randomObject = random.Random()
    randomObject.seed(randomSeed + key)

    startDate = datetime.datetime.strptime(configRow["Start Date"], "%m/%d/%Y")
    endDate = datetime.datetime.strptime(configRow["End Date"], "%m/%d/%Y")
    numdays = (endDate - startDate).days

    impressions = generate(
        randomObject,
        configRow["Publisher"],
        configRow["Advertiser"],
        configRow["Event Groups"],
        getPlatformDistSpec(configRow),
        getGenderDistSpec(configRow),
        getAgeDistSpec(configRow),
        getRealFreqDistSpec(configRow),
        startDate,
        numdays,
        configRow["Impressions"],
        configRow["Total Reach"] * configRow["Impressions"],
    )

    impressionsDataFrame = pd.DataFrame.from_records([asdict(imp) for imp in impressions])
    impressionsDataFrame.to_csv(f"{configRow['Publisher']}_row_{key}_fake_data.csv", mode="a", index=False)
    print(f"END {configRow}")


if __name__ == "__main__":
    args = parser.parse_args()
    edpName = args.edpname
    randomSeed = args.randomseed
    print("Random Seed = {},  Edp Name = {}".format(randomSeed, edpName))
    df = pd.read_csv("config.csv")
    df = df[df["Publisher"] == edpName]

    start = time.time()

    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Iterate over the rows in the dataframe
    for row in df.iterrows():
        # Submit the task to the pool
        pool.apply_async(generate_and_analyze_for_edp, args=(row[0], row[1], randomSeed))

    # Close the pool
    pool.close()

    # Wait for all tasks to complete
    pool.join()

    end = time.time()
    print("Elapsed time : ", (end - start))
