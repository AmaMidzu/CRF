import numpy as np

class CRF(object):

    def __init__(self, label_codebook, feature_codebook):
        self.label_codebook = label_codebook
        self.feature_codebook = feature_codebook
        self.num_labels = len(self.label_codebook)
        self.num_features = len(self.feature_codebook)
        self.feature_parameters = np.zeros((self.num_labels, self.num_features))
        self.transition_parameters = np.zeros((self.num_labels, self.num_labels))

    def train(self, training_set, dev_set):
        """Training function

        Feel free to adjust the hyperparameters (learning rate and batch sizes)
        """
        self.train_sgd(training_set, dev_set, 0.001, 200)

    def train_sgd(self, training_set, dev_set, learning_rate, batch_size):
        """Minibatch SGF for training linear chain CRF

        This should work. But you can also implement early stopping here
        i.e. if the accuracy does not grow for a while, stop.
        """

        num_batches = len(training_set) / batch_size
        total_expected_feature_count = np.zeros((self.num_labels, self.num_features))
        total_expected_transition_count = np.zeros((self.num_labels, self.num_labels))
        print 'With all parameters = 0, the accuracy is %s' % sequence_accuracy(self, dev_set)
        for i in xrange(10):
            for j in xrange(num_batches):
                batch = training_set[j*batch_size:(j+1)*batch_size]
                total_expected_feature_count.fill(0)
                total_expected_transition_count.fill(0)
                total_observed_feature_count, total_observed_transition_count = self.compute_observed_count(batch)

                for sequence in batch:
                    transition_matrices = self.compute_transition_matrices(sequence)
                    alpha_matrix = self.forward(sequence, transition_matrices)
                    beta_matrix = self.backward(sequence, transition_matrices)
                    expected_feature_count, expected_transition_count = self.compute_expected_feature_count(sequence, alpha_matrix, beta_matrix, transition_matrices) # exp counts updated
                    total_expected_feature_count += expected_feature_count
                    total_expected_transition_count += expected_transition_count

                feature_gradient = (total_observed_feature_count - total_expected_feature_count) / len(batch)
                transition_gradient = (total_observed_transition_count - total_expected_transition_count) / len(batch)

                self.feature_parameters += learning_rate * feature_gradient
                self.transition_parameters += learning_rate * transition_gradient
                print sequence_accuracy(self, dev_set)


    def compute_transition_matrices(self, sequence):
        """Compute transition matrices (denoted as M on the slides)

        Compute transition matrix M for all time steps.

        We add one extra dummy transition matrix at time 0.
        This one dummy transition matrix should not be used ever, but it
        is there to make the index consistent with the slides

        The matrix for the first time step does not use transition features
        and should be a diagonal matrix.

        TODO: Implement this function

        Returns :
            a list of transition matrices
        """
        transition_matrices = []
        # dummy matrix at 0
        transition_matrix = np.zeros((self.num_labels, self.num_labels))
        transition_matrices.append(transition_matrix)
        # diagonal matrix at 1
        transition_matrix = np.zeros((self.num_labels, self.num_labels))
        # loop to create the diagonal
        for i in xrange(self.num_labels):
            # HMM's p(q_t|q_t-1) and p(o_t|q_t)
            transition_matrix[i][i]=np.exp(self.feature_parameters[i][self.feature_codebook['T0='+sequence[0].features()[0][1]]])
        transition_matrices.append(transition_matrix)
        for t in xrange(1,len(sequence)):
            # compute transition matrix
            transition_matrix = np.zeros((self.num_labels, self.num_labels))
            for i in xrange(self.num_labels):
                for j in xrange(self.num_labels):
                    # exp(lam(y', y| X))
                    param=0
                    for feature in sequence[t].sequence_features(t, sequence):
                        param+=self.feature_parameters[j][self.feature_codebook[feature]]
                    transition_matrix[j][i] = np.exp(param + self.transition_parameters[j][i])
            transition_matrices.append(transition_matrix)
        return transition_matrices

    def forward(self, sequence, transition_matrices):
        """Compute alpha matrix in the forward algorithm

        TODO: Implement this function
        """
        alpha_matrix = np.zeros((self.num_labels, len(sequence) + 1))
        # alpha_0=[1]
        alpha_matrix[:,0]=1
        for t in xrange(1, len(sequence) + 1):
            for i in xrange(self.num_labels):
                k=0
                for j in xrange(self.num_labels):
                    k+=transition_matrices[t][j][i]*alpha_matrix[j][t-1]
                alpha_matrix[i][t]=k
        return alpha_matrix

    def backward(self, sequence, transition_matrices):
        """Compute beta matrix in the backward algorithm

        TODO: Implement this function
        """
        beta_matrix = np.zeros((self.num_labels, len(sequence) + 1))
        time = range(len(sequence))
        time.reverse()
        # beta_T=[1]
        beta_matrix[:,len(sequence)]=1
        for t in time:
            for i in xrange(self.num_labels):
                k=0
                for j in xrange(self.num_labels):
                    k+=transition_matrices[t+1][i][j]*beta_matrix[j][t+1]
                beta_matrix[i][t]=k
        return beta_matrix

    def decode(self, sequence):
        """Find the best label sequence from the feature sequence

        TODO: Implement this function

        Returns :
            a list of label indices (the same length as the sequence)
        """
        transition_matrices = self.compute_transition_matrices(sequence)
        alphas = self.forward(sequence, transition_matrices)
        betas = self.backward(sequence, transition_matrices)
        decoded_sequence = []
        normalized_score_of_label=np.zeros((self.num_labels, len(sequence)))
        for t in xrange(len(sequence)):
            for i in xrange(self.num_labels):
                normalized_score_of_label[i][t]=alphas[i][t]*betas[i][t]/sum([alphas[j][t]*betas[j][t] for j in xrange(self.num_labels)])
            decoded_sequence.append(np.argmax(normalized_score_of_label[:,t]))
        return decoded_sequence

    def compute_observed_count(self, sequences):
        """Compute observed counts of features from the minibatch

        This is implemented for you.

        Returns :
            A tuple of
                a matrix of feature counts
                a matrix of transition-based feature counts
        """

        feature_count = np.zeros((self.num_labels, self.num_features))
        transition_count = np.zeros((self.num_labels, self.num_labels))
        for sequence in sequences:
            for t in xrange(len(sequence)):
                if t > 0:
                    transition_count[sequence[t-1].label_index, sequence[t].label_index] += 1
                feature_count[sequence[t].label_index, sequence[t].feature_vector] += 1
        return feature_count, transition_count

    def compute_expected_feature_count(self, sequence,
            alpha_matrix, beta_matrix, transition_matrices):
        """Compute expected counts of features from the sequence

        This is implemented for you.

        Returns :
            A tuple of
                a matrix of feature counts
                a matrix of transition-based feature counts
        """

        feature_count = np.zeros((self.num_labels, self.num_features))
        sequence_length = len(sequence)
        Z = np.sum(alpha_matrix[:,-1])

        #gamma = alpha_matrix * beta_matrix / Z
        gamma = np.exp(np.log(alpha_matrix) + np.log(beta_matrix) - np.log(Z))
        for t in xrange(sequence_length):
            for j in xrange(self.num_labels):
                feature_count[j, sequence[t].feature_vector] += gamma[j, t]

        transition_count = np.zeros((self.num_labels, self.num_labels))
        for t in xrange(sequence_length - 1):
            transition_count += (transition_matrices[t] * np.outer(alpha_matrix[:, t], beta_matrix[:,t+1])) / Z
        return feature_count, transition_count

def sequence_accuracy(sequence_tagger, test_set):
    correct = 0.0
    total = 0.0
    for sequence in test_set:
        decoded = sequence_tagger.decode(sequence)
        assert(len(decoded) == len(sequence))
        total += len(decoded)
        for i, instance in enumerate(sequence):
            if instance.label_index == decoded[i]:
                correct += 1
    return correct / total
