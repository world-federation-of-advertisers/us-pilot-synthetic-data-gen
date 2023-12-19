from unittest import TestCase
from generate import generate
import random
import datetime
from pytest import approx
import collections


ACCEPTABLE_RELATIVE_ERROR = 1e-1


class GenerateTest(TestCase):
    def test_simple_spec(self):
        random_seed = 42
        randomObject = random.Random()
        randomObject.seed(random_seed)
        ageRangeDistSpec = [
            ("age_18_24", 0.2),
            ("age_25_34", 0.3),
            ("age_35_44", 0.1),
            ("age_45_54", 0.1),
            ("age_55_64", 0.2),
            ("age_65_plus", 0.1),
        ]
        genderDistSpec = [
            ("male", 0.3),
            ("female", 0.7),
        ]
        platformDistSpec = [
            ("desktop", 0.6),
            ("mobile", 0.4),
        ]
        maxFreq = 5
        realFreqDistSpec = [(1, 800), (2, 600), (3, 500), (4, 400), (maxFreq, 650)]
        startDate = datetime.date(2022, 9, 1)
        numdays = 91
        total_impressions = 9_000
        total_reach = 2_950
        impressions = generate(
            randomObject,
            "EDP1",
            "SomeMC",
            "SomeCampaign",
            platformDistSpec,
            genderDistSpec,
            ageRangeDistSpec,
            realFreqDistSpec,
            startDate,
            numdays,
            total_impressions,
            total_reach,
        )

        generated_number_of_impressions = len(impressions)

        # Assert generated number of impressions is correct
        self.assertTrue(
            generated_number_of_impressions == approx(total_impressions, rel=ACCEPTABLE_RELATIVE_ERROR),
        )

        # Assert generated reach is correct
        generated_reach = len(set(list(map(lambda x: x.vid, impressions))))
        self.assertTrue(generated_reach == approx(total_reach, rel=ACCEPTABLE_RELATIVE_ERROR))

        platformCountDict = dict(list(map(lambda x: (x[0], 0), platformDistSpec)))
        genderCountsDict = dict(list(map(lambda x: (x[0], 0), genderDistSpec)))
        ageRangeCountsDict = dict(list(map(lambda x: (x[0], 0), ageRangeDistSpec)))

        # Assert generated completion and viewibilty distributions are correct
        for imp in impressions:
            platformCountDict[imp.platform] += 1
            genderCountsDict[imp.gender] += 1
            ageRangeCountsDict[imp.ageRange] += 1

        for platformKey in platformCountDict:
            self.assertTrue(
                (platformCountDict[platformKey] / generated_number_of_impressions)
                == approx(dict(platformDistSpec)[platformKey], rel=ACCEPTABLE_RELATIVE_ERROR),
            )

        for genderKey in genderCountsDict:
            self.assertTrue(
                (genderCountsDict[genderKey] / generated_number_of_impressions)
                == approx(dict(genderDistSpec)[genderKey], rel=ACCEPTABLE_RELATIVE_ERROR),
            )

        for ageRangeKey in ageRangeCountsDict:
            self.assertTrue(
                (ageRangeCountsDict[ageRangeKey] / generated_number_of_impressions)
                == approx(dict(ageRangeDistSpec)[ageRangeKey], rel=ACCEPTABLE_RELATIVE_ERROR),
            )

        # Assert that Frequency Distribution is correct
        counter = collections.Counter(map(lambda x: x.vid, impressions))
        hist = dict(collections.Counter([i[1] for i in counter.items()]).items())
        if maxFreq + 1 in hist:
            hist[maxFreq] += hist[maxFreq + 1]
        # print(hist)
        realFreqDict = dict(realFreqDistSpec)
        for freqKey in realFreqDict:
            self.assertTrue(
                hist[freqKey] == approx(realFreqDict[freqKey], rel=2e-1),
            )
