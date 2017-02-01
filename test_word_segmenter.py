from corpus import Character, Character2, CharacterTest, ThaiWordCorpus
from crf import CRF, sequence_accuracy
from unittest import TestCase, main

class TestSegmenting(TestCase):

    def setUp(self):
        self.corpus = ThaiWordCorpus('orchid97_features.bio', Character)
        crf = CRF(self.corpus.label_codebook, self.corpus.feature_codebook)
        self.crf =crf

    def test_segmenting(self):
        train = self.corpus[0:20000]
        dev = self.corpus[20000:20050]
        test = self.corpus[21000:23000]
        self.crf.train(train, dev)

        accuracy = sequence_accuracy(self.crf, test)
        self.assertGreaterEqual(accuracy, 0.80)


if __name__ == '__main__':
    main(verbosity=2)


