#!/usr/bin/python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import pickle
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from os import remove
from copy import deepcopy
from operator import itemgetter
import numpy
try:
    from numpy import array
    from scipy import sparse
    from sklearn.datasets import load_svmlight_file
    from sklearn import svm
    from sklearn.linear_model import Perceptron
except ImportError:
    pass

from nltk.parse import ParserI, DependencyGraph, DependencyEvaluator
import time


class Configuration(object):
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """

    def __init__(self, dep_graph):
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        """
        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]  # The root element
        self.buffer = list(range(1, len(dep_graph.nodes)))  # The rest is in the buffer
        self.arcs = []  # empty set of arc
        self._tokens = dep_graph.nodes
        self._max_address = len(self.buffer)

    def __str__(self):
        return 'Stack : ' + \
            str(self.stack) + '  Buffer : ' + str(self.buffer) + '   Arcs : ' + str(self.arcs)

    def _check_informative(self, feat, flag=False):
        """
        Check whether a feature is informative
        The flag control whether "_" is informative or not
        """
        if feat is None:
            return False
        if feat == '':
            return False
        if flag is False:
            if feat == '_':
                return False
        return True

    def extract_features(self):
        """
        Extract the set of features for the current configuration. Implement standard features as describe in
        Table 3.2 (page 31) in Dependency Parsing book by Sandra Kubler, Ryan McDonal, Joakim Nivre.
        Please note that these features are very basic.
        :return: list(str)
        """
        result = []
        # Todo : can come up with more complicated features set for better
        # performance.
        if len(self.stack) > 0:
            # Stack 0
            stack_idx0 = self.stack[len(self.stack) - 1]
            token = self._tokens[stack_idx0]
            if self._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])
            #if 'lemma' in token and self._check_informative(token['lemma']):
            #    result.append('STK_0_LEMMA_' + token['lemma'])
            if self._check_informative(token['tag']):
                result.append('STK_0_POS_' + token['tag'])
            if 'feats' in token and self._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)
            # Stack 1
            if len(self.stack) > 1:
                stack_idx1 = self.stack[len(self.stack) - 2]
                token = self._tokens[stack_idx1]
                if self._check_informative(token['tag']):
                    result.append('STK_1_POS_' + token['tag'])
#adding
                #if 'feats' in token and self._check_informative(token['feats']):
                #    feats = token['feats'].split("|")
                #    for feat in feats:
                #        result.append('STK_1_FEATS_' + feat)
                        # Stack 1
            if len(self.stack) > 2:
                stack_idx2 = self.stack[len(self.stack) - 3]
                token = self._tokens[stack_idx1]
                if self._check_informative(token['tag']):
                    result.append('STK_2_POS_' + token['tag'])

            # Left most, right most dependency of stack[0]
            left_most = 1000000
            right_most = -1
            dep_left_most = ''
            dep_right_most = ''
            for (wi, r, wj) in self.arcs:
                if wi == stack_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if self._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)
            
##adding      # Left most, right most dependency of stack[1]
            if len(self.stack) > 1:
                left_most = 1000000
                right_most = -1
                dep_left_most = ''
                dep_right_most = ''
                for (wi, r, wj) in self.arcs:
                    if wi == stack_idx1:
                        if (wj > wi) and (wj > right_most):
                            right_most = wj
                            dep_right_most = r
                        if (wj < wi) and (wj < left_most):
                            left_most = wj
                            dep_left_most = r
                if self._check_informative(dep_left_most):
                    result.append('STK_1_LDEP_' + dep_left_most)
                if self._check_informative(dep_right_most):
                    result.append('STK_1_RDEP_' + dep_right_most)

#adding      # Left most, right most dependency of stack[1]
            #if len(self.stack) > 2:
            #    left_most = 1000000
            #    right_most = -1
            #    dep_left_most = ''
            #    dep_right_most = ''
            #    for (wi, r, wj) in self.arcs:
            #        if wi == stack_idx2:
            #            if (wj > wi) and (wj > right_most):
            #                right_most = wj
            #                dep_right_most = r
            #            if (wj < wi) and (wj < left_most):
            #                left_most = wj
            #                dep_left_most = r
            #    if self._check_informative(dep_left_most):
            #        result.append('STK_2_LDEP_' + dep_left_most)
            #    if self._check_informative(dep_right_most):
            #        result.append('STK_2_RDEP_' + dep_right_most)

        result2=[]
        # Check Buffered 0
        if len(self.buffer) > 0:
            # Buffer 0
            buffer_idx0 = self.buffer[0]
            token = self._tokens[buffer_idx0]
            if self._check_informative(token['word'], True):
                result2.append('BUF_0_FORM_' + token['word'])
            #if 'lemma' in token and self._check_informative(token['lemma']):
            #    result2.append('BUF_0_LEMMA_' + token['lemma'])
            if self._check_informative(token['tag']):
                result2.append('BUF_0_POS_' + token['tag'])
            if 'feats' in token and self._check_informative(token['feats']):
                feats = token['feats'].split("|")
                for feat in feats:
                    result2.append('BUF_0_FEATS_' + feat)
            # Buffer 1
            if len(self.buffer) > 1:
                buffer_idx1 = self.buffer[1]
                token = self._tokens[buffer_idx1]
                if self._check_informative(token['word'], True):
                    result2.append('BUF_1_FORM_' + token['word'])
                if self._check_informative(token['tag']):
                    result2.append('BUF_1_POS_' + token['tag'])
#adding
                #if 'feats' in token and self._check_informative(token['feats']):
                #    feats = token['feats'].split("|")
                #    for feat in feats:
                #        result2.append('BUF_1_FEATS_' + feat)
            if len(self.buffer) > 2:
                buffer_idx2 = self.buffer[2]
                token = self._tokens[buffer_idx2]
                if self._check_informative(token['tag']):
                    result2.append('BUF_2_POS_' + token['tag'])
            if len(self.buffer) > 3:
                buffer_idx3 = self.buffer[3]
                token = self._tokens[buffer_idx3]
                if self._check_informative(token['tag']):
                    result2.append('BUF_3_POS_' + token['tag'])
            # Left most, right most dependency of stack[0]
            left_most = 1000000
            right_most = -1
            dep_left_most = ''
            dep_right_most = ''
            for (wi, r, wj) in self.arcs:
                if wi == buffer_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result2.append('BUF_0_LDEP_' + dep_left_most)
            if self._check_informative(dep_right_most):
                result2.append('BUF_0_RDEP_' + dep_right_most)
#adding
            if len(self.buffer)>1:
                left_most = 1000000
                right_most = -1
                dep_left_most = ''
                dep_right_most = ''
                for (wi, r, wj) in self.arcs:
                    if wi == buffer_idx1:
                        if (wj > wi) and (wj > right_most):
                            right_most = wj
                            dep_right_most = r
                        if (wj < wi) and (wj < left_most):
                            left_most = wj
                            dep_left_most = r
                if self._check_informative(dep_left_most):
                    result2.append('BUF_1_LDEP_' + dep_left_most)
                if self._check_informative(dep_right_most):
                    result2.append('BUF_1_RDEP_' + dep_right_most)

            if len(self.buffer)>2:
                left_most = 1000000
                right_most = -1
                dep_left_most = ''
                dep_right_most = ''
                for (wi, r, wj) in self.arcs:
                    if wi == buffer_idx2:
                        if (wj > wi) and (wj > right_most):
                            right_most = wj
                            dep_right_most = r
                        if (wj < wi) and (wj < left_most):
                            left_most = wj
                            dep_left_most = r
                if self._check_informative(dep_left_most):
                    result2.append('BUF_2_LDEP_' + dep_left_most)
                if self._check_informative(dep_right_most):
                    result2.append('BUF_2_RDEP_' + dep_right_most)

        for idx in range(len(result)):
            for jdx in range(len(result2)):
                result.append(result[idx]+'_'+result2[jdx])
        result += result2
        result=list(set(result))
        return result


class Transition(object):
    """
    This class defines a set of transition which is applied to a configuration to get another configuration
    Note that for different parsing algorithm, the transition is different.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self, alg_option):
        """
        :param alg_option: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type alg_option: str
        """
        self._algo = alg_option
        if alg_option not in [
                TransitionParser.ARC_STANDARD,
                TransitionParser.ARC_EAGER]:
            raise ValueError(" Currently we only support %s and %s " %
                                        (TransitionParser.ARC_STANDARD, TransitionParser.ARC_EAGER))

    def left_arc(self, conf, relation):
        """
        Note that the algorithm for left-arc is quite similar except for precondition for both arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if (len(conf.buffer) <= 0) or (len(conf.stack) <= 0):
            return -1
        if conf.buffer[0] == 0:
            # here is the Root element
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]

        flag = True
        if self._algo == TransitionParser.ARC_EAGER:
            for (idx_parent, r, idx_child) in conf.arcs:
                if idx_child == idx_wi:
                    flag = False

        if flag:
            conf.stack.pop()
            idx_wj = conf.buffer[0]
            conf.arcs.append((idx_wj, relation, idx_wi))
        else:
            return -1

    def right_arc(self, conf, relation):
        """
        Note that the algorithm for right-arc is DIFFERENT for arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if (len(conf.buffer) <= 0) or (len(conf.stack) <= 0):
            return -1
        if self._algo == TransitionParser.ARC_STANDARD:
            idx_wi = conf.stack.pop()
            idx_wj = conf.buffer[0]
            conf.buffer[0] = idx_wi
            conf.arcs.append((idx_wi, relation, idx_wj))
        else:  # arc-eager
            idx_wi = conf.stack[len(conf.stack) - 1]
            idx_wj = conf.buffer.pop(0)
            conf.stack.append(idx_wj)
            conf.arcs.append((idx_wi, relation, idx_wj))

    def reduce(self, conf):
        """
        Note that the algorithm for reduce is only available for arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """

        if self._algo != TransitionParser.ARC_EAGER:
            return -1
        if len(conf.stack) <= 0:
            return -1

        idx_wi = conf.stack[len(conf.stack) - 1]
        flag = False
        for (idx_parent, r, idx_child) in conf.arcs:
            if idx_child == idx_wi:
                flag = True
        if flag:
            conf.stack.pop()  # reduce it
        else:
            return -1

    def shift(self, conf):
        """
        Note that the algorithm for shift is the SAME for arc-standard and arc-eager
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if len(conf.buffer) <= 0:
            return -1
        idx_wi = conf.buffer.pop(0)
        conf.stack.append(idx_wi)


class TransitionParser(ParserI):

    """
    Class for transition based parser. Implement 2 algorithms which are "arc-standard" and "arc-eager"
    """
    ARC_STANDARD = 'arc-standard'
    ARC_EAGER='arc-eager'
    def __init__(self, algorithm):
        """
        :param algorithm: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type algorithm: str
        """
        if not(algorithm in [self.ARC_STANDARD, self.ARC_EAGER]):
            raise ValueError(" Currently we only support %s and %s " %
                                        (self.ARC_STANDARD, self.ARC_EAGER))
        self._algorithm = algorithm

        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]

        if c_node['word'] is None:
            return None  # Root word

        if c_node['head'] == p_node['address']:
            return c_node['rel']
        else:
            return None

    def _convert_to_binary_features(self, features):
        """
        :param features: list of feature string which is needed to convert to binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is 'featureID:value' pairs
        """
        unsorted_result = []
        for feature in features:
            self._dictionary.setdefault(feature, len(self._dictionary))
            unsorted_result.append(self._dictionary[feature])
        #for idx in range(len(features)):
        #    for jdx in range(idx,len(features)):
        #        bi_f=features[idx]+'_'+features[jdx]
        #        if bi_f in self._dictionary:
        #            unsorted_result.append(self._dictionary[bi_f])
        #        else:
        #            self._dictionary.setdefault(bi_f, len(self._dictionary))
        #            unsorted_result.append(self._dictionary[bi_f])
        # Default value of each feature is 1.0
        return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))
    #this function, for example, gives out feautures's index, 11:'1.0' such than, refers the item in self._dictionary has 11 is chosen

    def _is_projective(self, depgraph):
        arc_list = []
        for key in depgraph.nodes:
            node = depgraph.nodes[key]           
            
            if 'head' in node:
                childIdx = node['address']
                parentIdx = node['head']
                if parentIdx is not None:
                    arc_list.append((parentIdx, childIdx))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key, binary_features, input_file):
        """
        write the binary features to input file and update the transition dictionary
        """
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key

        input_str = str(self._transition[key]) + ' ' + binary_features + '\n'
        #input_file.write(input_str)
        input_file.write(input_str.encode('utf-8'))
        ##the file ready to be trained
        #with open('C:/Users/haoran/Desktop/nlp/input_str','a') as f:
        #    f.writelines(input_str)


    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []
        total_features=[]        

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            count_proj += 1
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)

                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)   
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        # Get the max-index of buffer
                        maxID = conf._max_address

                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)
                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False

                        if precondition:
                            key = Transition.RIGHT_ARC + ':' + rel
                            self._write_to_file(
                                key,
                                binary_features,
                                input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(count_proj))
        #just try to save self._dictionary is ok
        pickle.dump(self._dictionary,open( "C:/Users/haoran/Desktop/nlp/self._dictionary", "wb" ) )
        pickle.dump(self._transition,open( "C:/Users/haoran/Desktop/nlp/self._transition", "wb" ) )
        pickle.dump(self._match_transition,open( "C:/Users/haoran/Desktop/nlp/self.match_transition", "wb" ) )
        print('model saved')
        #return training_seq
        return 0

    def train2(self,depgraphs,modelfile):
         
            #input_file = tempfile.NamedTemporaryFile(
            #    prefix='transition_parse.train',
            #    dir=tempfile.gettempdir(),
            #    delete=False)
            
            input_file = open('C:/Users/haoran/Desktop/nlp/dev_set/input_str.txt','wb')
            #self._dictionary=pickle.load(open('C:/Users/haoran/Desktop/nlp/self._dictionary', 'rb'))
            #self._transition=pickle.load(open( "C:/Users/haoran/Desktop/nlp/self._transition", "rb" ) )
            #self._match_transition=pickle.load(open( "C:/Users/haoran/Desktop/nlp/self.match_transition", "rb" ) )
            
            self._create_training_examples_arc_std(depgraphs, input_file)
            input_file.close()
            return 0

    def train(self, depgraphs, modelfile):
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        :param modelfile : file name to save the trained model
        :type modelfile : str
        """

        #try:
        #    input_file = tempfile.NamedTemporaryFile(
        #        prefix='transition_parse.train',
        #        dir=tempfile.gettempdir(),
        #        delete=False)

        #    if self._algorithm == self.ARC_STANDARD:
        #        self._create_training_examples_arc_std(depgraphs, input_file)
        try:
            #input_file = tempfile.NamedTemporaryFile(
            #    prefix='transition_parse.train',
            #    dir=tempfile.gettempdir(),
            #    delete=False)
            
            input_file = open('C:/Users/haoran/Desktop/nlp/dev_set/input_str.txt','rb')

            
            #self._create_training_examples_arc_std(depgraphs, input_file)
            input_file.close()
            x_train, y_train = load_svmlight_file(input_file.name)
            # The parameter is set according to the paper:
            # Algorithms for Deterministic Incremental Dependency Parsing by Joakim Nivre
            # Todo : because of probability = True => very slow due to
            # cross-validation. Need to improve the speed here
            ########################################### changing model####################
            #start=time.time()
            #model = svm.SVC(
            #    kernel='poly',
            #    degree=2,
            #    coef0=0,
            #    gamma=0.2,
            #    C=0.5,
            #    verbose=True,
            #    probability=True)         #try to change here, percepton
            #model.fit(x_train, y_train)
            #end=time.time()
            #print('simple svm :',str(end-start))
            #pickle.dump(model, open(modelfile, 'wb')) 
            ################################################################################

            start=time.time()
            model=Perceptron(penalty=None, alpha=0.0003, fit_intercept=True, n_iter=3, shuffle=True, verbose=0, eta0=1.0, n_jobs=1, random_state=7, class_weight=None, warm_start=True)
            model.fit(x_train, y_train)
            end=time.time()
        # Save the model to file name (as pickle)
            print('-----------------already fit ----------------')
            #pickle.dump(model, open(modelfile, 'wb'))
            joblib.dump(model,modelfile)
            print('perceptron time consume: '+str(end-start)) 
        finally:
            print('complete')
            remove(input_file.name)
            

    def parse(self, depgraphs, modelFile):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        cnt=0
        failure_lst=[]
        # First load the model
        model = joblib.load(modelFile)
        operation = Transition(self._algorithm)
        for depgraph in depgraphs:
            cnt+=1
            failure=0
            print('the present sentence ',cnt)
            print(failure_lst)
            conf = Configuration(depgraph)
            while len(conf.buffer) > 0 and failure!=1:
                features = conf.extract_features()
                #print(features)  #

                col = []
                row = []
                data = []
                for feature in features:
                    if feature in self._dictionary:
                        col.append(self._dictionary[feature])
                        row.append(0)
                        data.append(1.0)
                np_col = array(sorted(col))  # NB : index must be sorted
                np_row = array(row)
                np_data = array(data) 
                
            #attemp to writing log 
                #with open('C:/Users/haoran/Desktop/nlp/logging','w') as f:
                #    f.writelines('size of x_text'+str(len(features)))

                x_test = sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(self._dictionary)))
                #writting log
                #with open('C:/Users/haoran/Desktop/nlp/logging','a') as f:
                #    f.write(str(x_test.shape)+'\n')
                #    f.write(str(len(features))+'\n')
                #    f.write(str(len(self._dictionary)))
                # It's best to use decision function as follow BUT it's not supported yet for sparse SVM
                # Using decision funcion to build the votes array
                dec_func = model.decision_function(x_test)[0]
                #print(dec_func)
                votes = {}
                k = 0
                for i in range(len(model.classes_)):
                    for j in range(i+1, len(model.classes_)):
                        if  dec_func[k] > 0:
                            votes.setdefault(i,0)
                            votes[i] +=1
                        else:
                            votes.setdefault(j,0)
                            votes[j] +=1
                    k +=1
                 #Sort votes according to the values
                sorted_votes = sorted(votes.items(), key=itemgetter(1), reverse=True)

                ############################################################
                ## We will use predict_proba instead of decision_function
                #prob_dict = {}
                #pred_prob = model.predict_proba(x_test)[0]
                #print('here is x_test: ')
                #print(x_test)
                #print('here is pred_prob')
                #print(pred_prob,type(pred_prob))

                #for i in range(len(pred_prob)):
                #    prob_dict[i] = pred_prob[i]
                #sorted_Prob = sorted(
                #    prob_dict.items(),
                #    key=itemgetter(1),
                #    reverse=True)
                ##############################################################
                #print(sorted_votes,len(sorted_votes),'------length----')
                # Note that SHIFT is always a valid operation
                cnt2=0
                for (y_pred_idx, confidence) in sorted_votes:
                    cnt2+=1
                    if cnt2==len(sorted_votes):
                        print('meet a failure')
                        failure_lst.append(cnt)
                        failure=1
                        break
                    #y_pred = model.predict(x_test)[0]
                    # From the prediction match to the operation
                    y_pred = model.classes_[y_pred_idx]
                    #print('here is y_pred_index,y_pred ')
                    #print( y_pred_idx,y_pred)
                    #print(model.classes_)


                    if y_pred in self._match_transition:
                        strTransition = self._match_transition[y_pred]
                        baseTransition = strTransition.split(":")[0]
                        #print('---------------------y_pred-------------',y_pred)
                        #print(self._match_transition)
                        #print(strTransition,baseTransition)

                        if baseTransition == Transition.LEFT_ARC:
                            if operation.left_arc(conf, strTransition.split(":")[1]) != -1:
                                break
                        elif baseTransition == Transition.RIGHT_ARC:
                            if operation.right_arc(conf, strTransition.split(":")[1]) != -1:
                                break
                        elif baseTransition == Transition.REDUCE:
                            if operation.reduce(conf) != -1:
                                break
                        elif baseTransition == Transition.SHIFT:
                            if operation.shift(conf) != -1:
                                break
                    else:
                        raise ValueError("The predicted transition is not recognized, expected errors")

            # Finish with operations build the dependency graph from Conf.arcs
            if cnt2!=len(sorted_votes):
                new_depgraph = deepcopy(depgraph)
                for key in new_depgraph.nodes:
                    node = new_depgraph.nodes[key]
                    node['rel'] = ''
                    # With the default, all the token depend on the Root
                    node['head'] = 0                             #change here to zero-based?
                for (head, rel, child) in conf.arcs:
                    c_node = new_depgraph.nodes[child]
                    c_node['head'] = head
                    c_node['rel'] = rel
                result.append(new_depgraph)
            else :
                result.append(depgraph)

        return result
