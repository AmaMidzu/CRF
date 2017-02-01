from corpus import Character, Character2, CharacterTest, ThaiWordCorpus
from crf import CRF
from unittest import TestCase, main

class TestCRF(TestCase):

    def setUp(self):
        self.corpus = ThaiWordCorpus('orchid97_features.bio.small', Character)
        crf = CRF(self.corpus.label_codebook, self.corpus.feature_codebook)
        self.crf =crf

    def test_forward_backward(self):
        sequence = self.corpus[0]
        transition_matrices = self.crf.compute_transition_matrices(sequence)
        alpha = self.crf.forward(sequence, transition_matrices)
        beta = self.crf.backward(sequence, transition_matrices)
        Z = alpha[:,-1].sum()
        for t in range(len(sequence)-1):
            self.assertEqual(alpha[:,t].dot(beta[:,t]), Z)

    def test_alpha0(self):
        """ Check the base case for alpha"""
        sequence = self.corpus[0]
        transition_matrices = self.crf.compute_transition_matrices(sequence)
        alpha = self.crf.forward(sequence, transition_matrices)
        for li in range(alpha.shape[0]):
            self.assertEqual(alpha[li,0], 1)

    def test_betaT(self):
        """ Check the base case for beta"""
        sequence = self.corpus[0]
        transition_matrices = self.crf.compute_transition_matrices(sequence)
        beta = self.crf.backward(sequence, transition_matrices)
        for li in range(beta.shape[0]):
            self.assertEqual(beta[li, -1], 1)

if __name__ == '__main__':
    main(verbosity=2)
