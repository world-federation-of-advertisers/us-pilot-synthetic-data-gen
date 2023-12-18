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
import datetime
import numpy as np
from scipy import stats
import operator
import inspect
import csv
import math
from functools import reduce
import itertools


def get_virtual_people_buckets(bucket_probs):
    ranges = []
    start = 0
    for bucket, prob in bucket_probs:
        width = int(prob * NUM_VIRTUAL_PEOPLE)
        ranges.append((bucket, range(start, start + width)))
        start += width
    return ranges


# Total number of Virtual people, per the US population
NUM_VIRTUAL_PEOPLE = 300_000_000

# Daily uniform noise size as fraction of the impressions that day.
DAILY_NOISE_FACTOR = 0.1

# Acceptable change applied to max frequency to make the num impressions
# play well with freq distribution.
ACCEPTABLE_FREQ_DIST_CORRECTION_FACTOR = 0.1

RAND_MIN = 0
RAND_MAX = 100_000

POPULATION_GENDER_DIST = [("male", 0.5), ("female", 0.5)]
POPULATION_AGE_DIST = [
    ("age_18_24", 0.15),
    ("age_25_34", 0.25),
    ("age_35_44", 0.15),
    ("age_45_54", 0.15),
    ("age_55_64", 0.15),
    ("age_65_plus", 0.15),
]

# POPULATION_BUCKET_RANGES holds the ranges of vids for each combination of age and gender.
POPULATION_CARTESIAN = itertools.product(POPULATION_GENDER_DIST, POPULATION_AGE_DIST)
POPULATION_BUCKET_PROBS = [((elem[0][0], elem[1][0]), elem[0][1] * elem[1][1]) for elem in POPULATION_CARTESIAN]
POPULATION_BUCKET_RANGES = dict(get_virtual_people_buckets(POPULATION_BUCKET_PROBS))


class CampaignSpec:
    """Samples impressions on a given EDP on given dates, such that they approximately align with the given number of impressions, reach, and distributions of frequency, video completion, and viewability

    1. At initialization normalizes and reconstructs the frequency distribution to pad the frequency to meet the requirements of the other parameters.
        E.g. If freq distrubution is specified by [(1, 800), (2, 600), (3, 500)] 800 impressions with freq=1, 600 with freq=2...
             Then converts it to the prob distribution [(1, 0.42), (2, 0.31), (3, 0.27)]
             Given freq dist implies 1*800+ 2*600 + 3*500 = 4100 impressions but the given impresison requirement can be different (e.g. 5000)
                If so, then pads another frequency (4) to match that 4100+ 4*225 = 5000.
             Also makes sure that this correction does not change the given freq dist more than a small amount
    2. At initialization creates a pool of vids to be used in the sampling. This pool of vids consists of repeats specified by the frequency distribution.
        E.g : If the freq dist specifies 5 impressions with freq=1 and 3 with freq=2 using vidSet {1..100} we can generate
              vids=[1, 2, 3, 4, 5, 6 , 6, 7, 7, 8, 8] then, randomly shuffle this pool of vids.
    3. Selects number of impressions for each day = (total_impressions/numdays) + noise.  Where noise is uniform.
    3. For each day, pops vids from the pool according to the number of impressions for that day.
    4. For each impression, independently samples the video completion and viewability specified by the given distributions for them.
    """

    def __init__(
        self,
        edpId,
        mcId,
        cId,
        sd,
        nd,
        nImp,
        tr,
        freqDistSpec,
        platformDistSpec,
        genderDistSpec,
        ageDistSpec,
        randomObject,
    ):
        self.event_data_provider_id = edpId
        self.measurementConsumer_id = mcId
        self.campaign_id = cId
        self.num_days = nd
        self.dates = [sd + datetime.timedelta(days=x) for x in range(nd)]
        self.total_impressions = nImp
        self.total_reach = tr
        self.random = randomObject
        tempFreqDist = DiscreteDist(self.normalize(freqDistSpec), self.random.randint(RAND_MIN, RAND_MAX))
        self.freq_dist = self.reconstruct_freq_dist(tempFreqDist)
        self.platform_dist = DiscreteDist(platformDistSpec, self.random.randint(RAND_MIN, RAND_MAX))

        self.gender_dist = DiscreteDist(genderDistSpec, self.random.randint(RAND_MIN, RAND_MAX))
        self.age_dist = DiscreteDist(ageDistSpec, self.random.randint(RAND_MIN, RAND_MAX))

        # campaign_bucket_amounts holds the amoung of virtual people for each combination of age and gender for this campaign.
        campaign_cartesian = list(itertools.product(genderDistSpec, ageDistSpec))
        self.campaign_bucket_amounts = [
            ((elem[0][0], elem[1][0]), int(elem[0][1] * elem[1][1] * self.total_reach)) for elem in campaign_cartesian
        ]
        self.virtual_people = self.sampleVirtualPeople()

    def normalize(self, freqDistSpec):
        temp_normailized = [(val, round(reach / self.total_reach, 3)) for (val, reach) in freqDistSpec]
        max_freq = max([val for (val, prob) in temp_normailized])
        prob_for_max_freq = list(filter(lambda x: x[0] == max_freq, temp_normailized))[0][1]

        distButMax = list(filter(lambda x: x[0] != max_freq, temp_normailized))
        implied_prob_for_max_freq = round(1 - sum([prob for (val, prob) in distButMax]), 3)

        # There can be a correction but not much
        assert implied_prob_for_max_freq >= prob_for_max_freq
        assert (implied_prob_for_max_freq - prob_for_max_freq) < ACCEPTABLE_FREQ_DIST_CORRECTION_FACTOR

        normailized = distButMax + [(max_freq, implied_prob_for_max_freq)]
        return normailized

    def reconstruct_freq_dist(self, freqDist):
        max_freq = max([val for (val, prob) in freqDist.prob_tuples])

        prob_for_max_freq = list(filter(lambda x: x[0] == max_freq, freqDist.prob_tuples))[0][1]

        implied_number_of_impressions = sum(
            [self.total_reach * prob * val for (val, prob) in freqDist.prob_tuples if val != max_freq]
        )

        remaining_number_of_impressions = self.total_impressions - implied_number_of_impressions
        reach_in_max_freq = self.total_reach * prob_for_max_freq
        new_max_freq = math.ceil(remaining_number_of_impressions / reach_in_max_freq)
        new_prob_tuples = list(filter(lambda x: x[0] != max_freq, freqDist.prob_tuples)) + [
            (new_max_freq, prob_for_max_freq)
        ]
        new_freq_dist = DiscreteDist(new_prob_tuples, freqDist.seed)

        print(
            "Changed the old max frequency",
            max_freq,
            "to a new max frequency ==> ",
            new_max_freq,
        )
        return new_freq_dist

    def sampleImpressionsForDay(self, date):
        impressions = []
        num_impressions_this_day = int(
            (self.total_impressions / float(self.num_days))
            * self.random.uniform(1 - DAILY_NOISE_FACTOR, 1 + DAILY_NOISE_FACTOR)
        )
        for i in range(num_impressions_this_day):
            virtual_person = self.virtual_people.pop()
            imp = Impression(
                self.event_data_provider_id,
                self.campaign_id,
                self.measurementConsumer_id,
                virtual_person.vid,
                virtual_person.gender,
                virtual_person.age,
                self.platform_dist.sample(),
                date.strftime("%d-%m-%Y"),
            )
            impressions.append(imp)
        return impressions

    # Sampled virtual people to fit the freq_dist, total_impressions and reach requirements
    def sampleVirtualPeople(self):
        padding_factor = 1 + (DAILY_NOISE_FACTOR)
        virtual_people_to_use = []
        for bucket, amount in self.campaign_bucket_amounts:
            # Multiplied by the padding factor so that we don't run out of virtualPeople to sample due to daily reach noises adding up
            amount_to_use = int(padding_factor * amount)
            virtual_people_to_use += [
                VirtualPerson(vid, bucket[0], bucket[1])
                for vid in self.random.sample(POPULATION_BUCKET_RANGES[bucket], amount_to_use)
            ]
        virtual_people = []
        while len(virtual_people_to_use) > 0:  # Keep generating until you run out of virtualPeople used for sampling
            num_impressions_for_virtual_person = self.freq_dist.sample()
            virtual_person = virtual_people_to_use.pop()
            virtual_person_replicated = [virtual_person] * num_impressions_for_virtual_person
            virtual_people += virtual_person_replicated
        self.random.shuffle(virtual_people)
        return virtual_people


@dataclass
class VirtualPerson:
    # id of the virtual person
    vid: int

    # Gender of the virtual person
    gender: str

    # Age bucket of the virtual person
    age: str


@dataclass
class Impression:
    """Class that represents a single impression."""

    # Id of the Event Data Provider
    event_data_provider_id: str

    # Id of the campaign this impression belongs to
    campaign_id: str

    # Id of the Measurement Consumer this impression belongs to
    mc_id: str

    # id of the virtual person that genereated this impression
    vid: int

    # Gender of the virtual person that genereated this impression
    gender: str

    # Age bucket of the virtual person that genereated this impression
    ageRange: str

    # Platform this impression was generated on
    platform: str  # dektop, mobile.

    # Date this impression happened
    date: str  # of the '%d-%m-%Y'


class DiscreteDist:
    """Class that represents Discrete distribution."""

    def __init__(self, prob_tuples, random_seed):
        self.seed = random_seed
        self.prob_tuples = prob_tuples
        self.vals = list(map(operator.itemgetter(0), prob_tuples))

        # Values must be unique
        assert len(set(self.vals)) == len(prob_tuples)

        self.probs = np.arange(len(prob_tuples)), list(map(operator.itemgetter(1), prob_tuples))
        self.custm = stats.rv_discrete(name="custm", values=self.probs, seed=self.seed)

    def sample(self):
        return self.vals[self.custm.rvs(size=1)[0]]

    def __str__(self):
        return str(self.prob_tuples)


class NoOpDiscreteDist(DiscreteDist):
    def __init__(self):
        super().__init__([(0, 1)], 0)

    def sample(self):
        return "NaN"


def generate(
    randomObject,
    edpId,
    mcId,
    campaignId,
    platformDistSpec,
    genderDistSpec,
    ageDistSpec,
    realFreqDistSpec,
    startdate,
    numdays,
    total_impressions,
    total_reach,
):
    campaignSpec = CampaignSpec(
        edpId,
        mcId,
        campaignId,
        startdate,
        numdays,
        total_impressions,
        total_reach,
        realFreqDistSpec,
        platformDistSpec,
        genderDistSpec,
        ageDistSpec,
        randomObject,
    )
    return reduce(
        list.__add__,
        [campaignSpec.sampleImpressionsForDay(date) for date in campaignSpec.dates],
    )
