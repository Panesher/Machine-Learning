import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    fv, tv = np.array(feature_vector), np.array(target_vector)
    if np.unique(fv).shape[0] < 2:
        return np.zeros(1), np.zeros(1), 0, 0
    
    idx = np.argsort(fv)
    feat_vec_sorted = fv[idx]
    target_sorted = tv[idx]
    elements_not_equal = feat_vec_sorted[1:] != feat_vec_sorted[:-1]

    thresholds = ((feat_vec_sorted[1:] + feat_vec_sorted[:-1]) / 2)[elements_not_equal]
    
    cnt_l = np.arange(1, feat_vec_sorted.shape[0])[elements_not_equal]
    cnt_r = feat_vec_sorted.shape[0] - cnt_l
    target_prefix = np.cumsum(target_sorted)[:-1][elements_not_equal] / cnt_l
    target_z = (np.cumsum(target_sorted[::-1][:-1]) / np.arange(
        1, feat_vec_sorted.shape[0]))[::-1][elements_not_equal]
    H_r = 1 - (1 - target_z) ** 2 - target_z ** 2
    H_l = 1 - (1 - target_prefix) ** 2 - target_prefix ** 2
    
    Q = -(cnt_l * H_l + cnt_r * H_r) / fv.shape[0]
    mx = Q.argmax()
    
    return thresholds, Q, thresholds[mx], Q[mx]


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        
    # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/base.py#L188
    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, '_' + key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out
    
    def _check_stop_condition(self, depth, samples):
        if self._max_depth is not None and depth >= self._max_depth:
            return True
        if self._min_samples_split is not None and samples < self._min_samples_split:
            return True
        return False

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._check_stop_condition(depth, sub_y.shape[0]):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if np.unique(feature_vector).shape[0] < 2:
                continue

            # TODO: make for min_samples_leaf
            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']
        
        feature_split = self._feature_types[node['feature_split']]
        if feature_split == 'categorical':
            child = node['left_child'] if x[node['feature_split']]  in node['categories_split'] else node['right_child']
        elif feature_split == 'real':
            child = node['left_child'] if x[node['feature_split']] < node['threshold'] else node['right_child']
        else:
            raise ValueError
        
        return self._predict_node(x, child)

    def fit(self, X, y):
        self._fit_node(np.array(X), np.array(y), self._tree)

    def predict(self, X_):
        predicted = []
        X = np.array(X_)
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
