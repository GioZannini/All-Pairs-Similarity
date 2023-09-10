from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, vstack
from tqdm import tqdm


class AllPairsSimilarityMaster:

    def __init__(self, docs_list, threshold):
        # set threshold
        self.threshold = threshold
        # number of docs
        self.n_docs = None
        # contain pair of (doc_id, tf_idf value)
        self.tfidf_docs = None
        # is a dictionary where {doc_key: list of id docs that have similarity over the threshold}
        self.similarity_map = {}
        # extract list of docs id
        self.keys = list(docs_list.keys())
        # calculate number of docs
        self.n_docs = len(self.keys)
        # extract doc's text
        docs = docs_list.values()
        # istance of TF-IDF
        TFIDF_transformer = TfidfVectorizer(strip_accents='unicode')
        # train algorithm
        TFIDF_transformer.fit(docs)
        # transform all docs in TF-IDF and create a csr matrix
        self.tfidf_computation = TFIDF_transformer.transform(docs)

    # measure similarity between pairs of documents
    def measure_similarity(self):
        pass

    # return similarity dictionary
    def get_similarity_map(self):
        return self.similarity_map
    
    # determine number of pairs
    def get_number_pairs(self):
        tmp_l = list(self.get_similarity_map().items())
        pairs = []
        for (k, l) in tmp_l:
            # determine number of pairs
            # v is a tupla (doc_id_2, cosine)
            # this create a list of tuple where the order is rispected
            # in this way I can determine the number of different pairs
            pairs += [(k, v[0]) if k<v[0] else (v[0], k) for v in l]
        return len(set(pairs))
    

class AllPairsSimilaritySeqVersion(AllPairsSimilarityMaster):

    def __init__(self, docs_list, threshold):
        super().__init__(docs_list, threshold)
        # create a list of pairs (id, tf-idf vector)
        self.tfidf_docs = [(ids, tfidf) for (ids, tfidf) in zip(self.keys, self.tfidf_computation)]
        # sort list of tuple in descendent way considering the number of element not equal zero
        # (optimization for reducing computational cost)
        self.tfidf_docs = sorted(self.tfidf_docs, reverse=True, key=AllPairsSimilaritySeqVersion.__key_order)
        # computation of minimum number of elements that must present in a vector to outdo threshold for doc i
        self.min_number_to_outdo_threshold = [self.__find_min_dot_prod_value(i) for i in range(self.n_docs)]

    # define order for sorting
    @staticmethod
    def __key_order(d):
        # extract sparse vector tf-idf
        sparse_vector_tfidf = d[1]
        # count how many element are different from zero
        # in vector
        return (sparse_vector_tfidf.nonzero()[1]).shape[0]

    # determine min number of values that are necessary to overtake the threshold
    def __find_min_dot_prod_value(self, n):
        # sort in decresing order by tf-idf value
        sorted_v = np.sort(self.tfidf_docs[n][1].toarray().flat)[::-1]
        # pre computation of dot product between it self
        pre_dot_prod = 0
        # find min number of element necessary to overtake threshold
        for n_min_values in range(sorted_v.shape[0]):
            # compute dot product of itself for each element in vector
            pre_dot_prod += sorted_v[n_min_values] ** 2
            if pre_dot_prod >= self.threshold:
                break
        return n_min_values + 1

    # measure similarity between pairs of documents
    def measure_similarity(self):
        # compute similarity
        # with this 2 for I only compute half of the similarity matrix
        for first_n in tqdm(range(self.n_docs), ncols=47):
            # determine how many values are necessary to overcome the threshold
            n_min_values = self.min_number_to_outdo_threshold[first_n]
            for second_n in range(first_n + 1, self.n_docs):
                # condition that indicates that no one vector can overtake threshold
                if (self.tfidf_docs[second_n][1].nonzero()[1]).shape[0] < n_min_values:
                    break
                # calculate similarity
                cosine_sim = self.tfidf_docs[first_n][1].dot(self.tfidf_docs[second_n][1].transpose())
                # extract similarity value
                cosine_sim = cosine_sim.toarray().flat[0]
                # if similarity is grater than threshold save pair
                if cosine_sim > self.threshold:
                    # populate list of first doc vector
                    self.similarity_map[self.tfidf_docs[first_n][0]] = self.similarity_map.get(
                        self.tfidf_docs[first_n][0], []) + [(self.tfidf_docs[second_n][0], cosine_sim)]
                    # populate list of second doc vector
                    self.similarity_map[self.tfidf_docs[second_n][0]] = self.similarity_map.get(
                        self.tfidf_docs[second_n][0], []) + [(self.tfidf_docs[first_n][0], cosine_sim)]


class AllPairsSimilaritySparkVersionBase(AllPairsSimilarityMaster):

    def __init__(self, docs_list, threshold, spark, compute_tf_idf_docs=True, partitions=None):
        super().__init__(docs_list, threshold)
        # spark context
        self.spark = spark
        # n° partition split for RDD
        self.partitions = partitions
        # decide if computing or not the tf_idf, is useful for subclass to avoid useless computation
        if compute_tf_idf_docs:
            # create a list of pairs (id, tf-idf vector)
            self.tfidf_docs = [(ids, tfidf) for (ids, tfidf) in zip(self.keys, self.tfidf_computation)]

    @staticmethod
    # (doc_id, tf_idf vector) -> (term_id, (doc_id, tf_idf vector))
    def _first_map_flat_func(x):
        return [(term_id, (x[0], x[1])) for term_id in x[1].nonzero()[1]]

    @staticmethod
    # compute the cosine similarity between docs
    def _second_map_func(x, threshold):
        # values of key
        l_pairs = x[1]
        # returned vector
        cosine_vector = []
        # num docs
        num_docs = len(l_pairs)
        for i_1 in range(num_docs):
            for i_2 in range(i_1 + 1, num_docs):
                # calculate similarity
                cosine_sim = l_pairs[i_1][1].dot(l_pairs[i_2][1].transpose())
                # extract similarity value
                cosine_sim = cosine_sim.toarray().flat[0]
                # if similarity is grater than threshold save pair
                if cosine_sim > threshold:
                    # thanks to the following operations I can create a for each doc_id
                    # the docs id that outdoes threshold
                    # populate list of first doc vector
                    cosine_vector.append((l_pairs[i_1][0], l_pairs[i_2][0], cosine_sim))
                    # populate list of second doc vector
                    cosine_vector.append((l_pairs[i_2][0], l_pairs[i_1][0], cosine_sim))
        return (x[0], cosine_vector)

    # measure similarity between pairs of documents
    def measure_similarity(self):
        threshold = self.threshold
        # RDD composed by pairs of (doc_id, tf_idf vector)
        rdd_tfidf = self.spark.parallelize(self.tfidf_docs, self.partitions)
        # execute map flat function
        mapped = rdd_tfidf.flatMap(AllPairsSimilaritySparkVersionBase._first_map_flat_func)
        # execute group by key and I map values as list
        group_by_key = mapped.groupByKey().mapValues(list)
        # execute cosine similarity for each key (term_id) in independent way
        cosines = group_by_key.map(lambda x: AllPairsSimilaritySparkVersionBase._second_map_func(x, threshold))
        # extract only values for each key
        cosines_values = cosines.values()
        # flat vectors for removing duplicates
        flat_cosines_values = cosines_values.flatMap(lambda x: x)
        # remove duplicates
        cosines_no_duplicates = flat_cosines_values.distinct()
        # create new map (doc_id_1, (doc_id_2, cosine))
        mapped_doc_keys = cosines_no_duplicates.map(lambda x: (x[0], (x[1], x[2])))
        # execute group by key and map values as list
        group_by_doc_key = mapped_doc_keys.groupByKey().mapValues(list)
        # save the final dictionary
        self.similarity_map = dict(group_by_doc_key.collect())


class AllPairsSimilaritySparkVersion2(AllPairsSimilaritySparkVersionBase):

    def __init__(self, docs_list, threshold, spark, partitions=None):
        super().__init__(docs_list, threshold, spark, False, partitions)
        # compute tfidf mean for each term along all docs and flat matrix
        tf_idf_mean_terms = self.tfidf_computation.mean(axis=0).A1
        # sort in decreasing order
        index_decresing_sort = np.argsort(tf_idf_mean_terms)[::-1]
        # order matrix by mean frequency
        self.ordered_tfidf_computation = self.tfidf_computation[::, index_decresing_sort]
        # adjust indices due to previous sorting
        self.__adjust_indices()
        # vector d* (vector containing max value for each term)
        d_star = self.ordered_tfidf_computation.max(axis=0).tocsr(copy=False)
        # number of terms
        n_terms = self.ordered_tfidf_computation.shape[1]
        # list containing boundaries for each doc
        boundary_docs = AllPairsSimilaritySparkVersion2.__find_boundaries(self.ordered_tfidf_computation, d_star,
                                                                          self.n_docs, n_terms, self.threshold)
        # create a list of pairs (id, tf-idf vector, boundary)
        self.tfidf_docs = [(ids, tfidf, b) for (ids, tfidf, b) in
                           zip(self.keys, self.ordered_tfidf_computation, boundary_docs)]

    # adjust indices after sorted by frequency
    def __adjust_indices(self):
        # empty csr
        final_csr = dok_matrix((0, self.ordered_tfidf_computation[0, ::].shape[1]))
        # correct indices in csr
        for i in range(self.n_docs):
            adjust_csr = csr_matrix(self.ordered_tfidf_computation[i, ::].toarray())
            final_csr = vstack((final_csr, adjust_csr))
        self.ordered_tfidf_computation = final_csr.tocsr()

    @staticmethod
    # find min value of terms necessary to outdo the threshold
    def __find_boundaries(ordered_tfidf, d_star, n_docs, n_terms, threshold):
        boundary_docs = []
        for d in range(n_docs):
            # watch only indices with values different from zero
            for t in ordered_tfidf[d, ::].indices:
                # calculate partial cosine
                partial_cosine = ordered_tfidf[d, 0:t + 1].dot(d_star[0, 0:t + 1].transpose())[0, 0]
                if partial_cosine >= threshold:
                    # this indicates the start to find a term different from zero to outdo threshold
                    boundary_docs.append(t)
                    # change doc
                    break
        return boundary_docs

    @staticmethod
    # find pos of val in vector
    def __binary_search(vector, f, t, val):
        # nearest value
        if f == t:
            return f + 1 if vector[f] < val else f
        mid = (t + f) // 2
        if vector[mid] < val:
            return AllPairsSimilaritySparkVersion2.__binary_search(vector, mid + 1, t, val)
        elif vector[mid] > val:
            return AllPairsSimilaritySparkVersion2.__binary_search(vector, f, mid - 1, val)
        else:
            return mid + 1 if vector[mid] < val else mid

    @staticmethod
    # (doc_id, tf_idf vector, boundary) -> (term_id, (doc_id, tf_idf vector))
    def _first_map_flat_func(x):
        vector = []
        # list of terms not equal zero
        terms_list = x[1].nonzero()[1]
        # find initial point in terms_list to watch
        start_point = AllPairsSimilaritySparkVersion2.__binary_search(terms_list, 0, terms_list.shape[0] - 1, x[2])
        # populate vector
        return [(terms_list[pos], (x[0], x[1])) for pos in range(start_point, terms_list.shape[0])]
    
    # measure similarity between pairs of documents
    def measure_similarity(self):
        threshold = self.threshold
        # RDD composed by pairs of (doc_id, tf_idf vector)
        rdd_tfidf = self.spark.parallelize(self.tfidf_docs, self.partitions)
        # execute map flat function
        mapped = rdd_tfidf.flatMap(AllPairsSimilaritySparkVersion2._first_map_flat_func)
        # execute group by key and I map values as list
        group_by_key = mapped.groupByKey().mapValues(list)
        # execute cosine similarity for each key (term_id) in independent way
        cosines = group_by_key.map(lambda x: AllPairsSimilaritySparkVersionBase._second_map_func(x, threshold))
        # extract only values for each key
        cosines_values = cosines.values()
        # flat vectors for removing duplicates
        flat_cosines_values = cosines_values.flatMap(lambda x: x)
        # remove duplicates
        cosines_no_duplicates = flat_cosines_values.distinct()
        # create new map (doc_id_1, (doc_id_2, cosine))
        mapped_doc_keys = cosines_no_duplicates.map(lambda x: (x[0], (x[1], x[2])))
        # execute group by key and map values as list
        group_by_doc_key = mapped_doc_keys.groupByKey().mapValues(list)
        # save the final dictionary
        self.similarity_map = dict(group_by_doc_key.collect())


class AllPairsSimilaritySparkVersion3(AllPairsSimilaritySparkVersion2):
    def __init__(self, docs_list, threshold, spark, partitions=None):
        super().__init__(docs_list, threshold, spark, partitions)

    @staticmethod
    # control if this term is the last contained inside tdf_idf of d1 and d2
    def __term_is_max(t, tfidf_d1, tfidf_d2):
        s1 = set(tfidf_d1[0, ::].indices)
        s2 = set(tfidf_d2[0, ::].indices)
        s = s1.intersection(s2)
        return max(s) == t

    @staticmethod
    # compute the cosine similarity between docs
    def _second_map_func(x, threshold):
        l_pairs = x[1]
        cosine_vector = []
        num_docs = len(l_pairs)
        for i_1 in range(num_docs):
            for i_2 in range(i_1 + 1, num_docs):
                if AllPairsSimilaritySparkVersion3.__term_is_max(x[0], l_pairs[i_1][1], l_pairs[i_2][1]):
                    # calculate similarity
                    cosine_sim = l_pairs[i_1][1].dot(l_pairs[i_2][1].transpose())
                    # extract similarity value
                    cosine_sim = cosine_sim.toarray().flat[0]
                    # if similarity is grater than threshold save pair
                    if cosine_sim > threshold:
                        # thanks to the following operations I can create a for each doc_id
                        # the docs id that outdoes threshold
                        # populate list of first doc vector
                        cosine_vector.append((l_pairs[i_1][0], l_pairs[i_2][0], cosine_sim))
                        # populate list of second doc vector
                        cosine_vector.append((l_pairs[i_2][0], l_pairs[i_1][0], cosine_sim))
        return (x[0], cosine_vector)
    
    # measure similarity between pairs of documents
    def measure_similarity(self):
        threshold = self.threshold
        # RDD composed by pairs of (doc_id, tf_idf vector)
        rdd_tfidf = self.spark.parallelize(self.tfidf_docs, self.partitions)
        # execute map flat function
        mapped = rdd_tfidf.flatMap(AllPairsSimilaritySparkVersion2._first_map_flat_func)
        # execute group by key and I map values as list
        group_by_key = mapped.groupByKey().mapValues(list)
        # execute cosine similarity for each key (term_id) in independent way
        cosines = group_by_key.map(lambda x: AllPairsSimilaritySparkVersion3._second_map_func(x, threshold))
        # extract only values for each key
        cosines_values = cosines.values()
        # flat vectors for removing duplicates
        flat_cosines_values = cosines_values.flatMap(lambda x: x)
        # remove duplicates
        cosines_no_duplicates = flat_cosines_values.distinct()
        # create new map (doc_id_1, (doc_id_2, cosine))
        mapped_doc_keys = cosines_no_duplicates.map(lambda x: (x[0], (x[1], x[2])))
        # execute group by key and map values as list
        group_by_doc_key = mapped_doc_keys.groupByKey().mapValues(list)
        # save the final dictionary
        self.similarity_map = dict(group_by_doc_key.collect())