import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix

def find_best_split(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    
    features_uniqs, value_count = np.unique(feature_vector, return_counts=True)
    if features_uniqs.shape[0] == 1:
        return None, None, -100500, -100500
    
    thresholds = (features_uniqs[0:-1] + features_uniqs[1:]) / 2
        
    n_features = feature_vector.shape[0]
    
    '''
        Мысля: t_0 < ... < t_m - неповторяющиеся пороги
        Рассматриваем правую ветвь ( x >= t )
        При увеличении t_i -> t_i+1 в ЛЕВУЮ ветвь попадает еще максимум одно *уникальное* значение признака
        С правой ветвью - ситуация противоположная
        В этом случае можно рассматривать в качестве порогов значения признака
        Это значит, что при увеличении i число объектов положительного класса в правой ветви будет не уменьшаться => cumsum
        
        Отмечу, что в моем способе формирования порогов \forall i  a < t_i < b, где a, b = min, max признака
        Бонусом идет то, что в каждой ветви будет как минимум одно значение признака (при нормальных inputs)
        
        Все рассмотрение будет строиться на объектах положительного класса
    '''
    
    # Нужно посчитать для каждого значения признака, сколько раз оно соответствует положительному классу
    # По сути, обычная агрегация, которую можно сделать через groupby -> sum
    fv_1, cnts_1 = np.unique(feature_vector[np.argwhere(target_vector == 1)], return_counts=True)
    fv_0, cnts_0 = np.unique(feature_vector[np.argwhere(target_vector == 0)], return_counts=True)
    lost = np.setdiff1d(fv_0, fv_1)
    fv_1 = np.append(fv_1, lost)
    cnts_1 = np.append(cnts_1, np.zeros(lost.shape[0]))
    inds = np.argsort(fv_1)

    quantity_positive = cnts_1[inds][:-1].astype(int) # Срезаем последнее, так как оно все равно всегда будет в правой ветви
    counts_positive_l = np.cumsum(quantity_positive)
    counts_positive_r = np.sum(target_vector) - counts_positive_l 
    
    counts_l = np.cumsum(value_count)[:-1]
    counts_r = n_features - counts_l
    
    def H(counts_positive, counts):
        return 1 - (counts_positive / counts) ** 2 - (1 - counts_positive / counts) ** 2
    
    ginis = - counts_l / n_features * H(counts_positive_l, counts_l) - \
              counts_r / n_features * H(counts_positive_r, counts_r)

    ind = np.argmax(ginis)
    gini_best = ginis[ind]
    threshold_best = thresholds[ind]
    
    return thresholds, ginis, threshold_best, gini_best

# have some problems
def find_best_split_old(feature_vector, target_vector):
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит (значение порога).
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    
    features_sorted = np.unique(feature_vector)
    thresholds = (features_sorted[0:-1] + features_sorted[1:]) / 2
    
    n_thresholds = thresholds.shape[0]
    n_features = feature_vector.shape[0]
    tr_mat = np.split(thresholds, indices_or_sections=n_thresholds)
    
    indicator_mat = csr_matrix((feature_vector >= tr_mat).astype(int))
    counts_r = np.sum(indicator_mat.A, axis=1)
    counts_l = n_features - counts_r
    
    counts_positive_r = np.sum(indicator_mat.A * target_vector, axis=1 )
    counts_positive_l = np.sum((1 - indicator_mat.A) * target_vector, axis=1 )
    
    def H(counts_positive, counts):
        return 1 - (counts_positive / counts) ** 2 - (1 - counts_positive / counts) ** 2
    
    ginis = - counts_l / n_features * H(counts_positive_l, counts_l) - \
              counts_r / n_features * H(counts_positive_r, counts_r)
    
    ind = np.argmax(ginis)
    gini_best = ginis[ind]
    threshold_best = thresholds[ind]
    
    return thresholds, ginis, threshold_best, gini_best

class DecisionTree:
    def __init__(self, feature_types: list, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types  # list номер признака - тип
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):

        # критерий: все объекты одного класса
        if np.all(sub_y == sub_y[0]):  # было != (критерий: все объекты одного класса)
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):  # было range(1, sub_X.shape[1])
            feature_type = self._feature_types[feature]
            categories_map = {}
            
            feature_values = sub_X[:, feature]
            
            if feature_type == "real":
                feature_vector = feature_values
            elif feature_type == "categorical":
                counts = Counter(feature_values)  # словарь: счетчик вхождений каждого значения у признака feature
                clicks = Counter(sub_X[sub_y == 1, feature])  # словарь: счетчик вхождений каждого значения с "+"-классом
                
                # выявляем соотношение для объектов "+"-класса
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # было деление наоборот
                    
                # сортируем категории по соотношению
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))  # в первом иксе было [1]
                # "индексация" категорий
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                # кодирование категориальных признаков
                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], feature_values))
                )  # добавил list() - без него возвращается итератор
                
            else:
                raise ValueError

            if len(feature_vector) == 0:  # было 3
                continue

            _, _, threshold, gini = find_best_split(feature_vector, np.array(sub_y))  # вместо sub_y - np.array
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":  # было с заглавной буквы
                    # выбор лучшего порога с точки зрения незакодированных значений
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError
        
        # обозначаем терминальность и берем самый частый класс
        # [0][0] добавлено
        if gini_best == -100500 or feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        else:
            node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
            
        # определяем дочерние узлы и вызываем функцию от них рекурсивно
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])  # sub_y[split] -> np.logical_not(split)

    def _predict_node(self, x, node, ubound=200):
        bound_counter = 0
        current_node = node
        while (current_node["type"] != "terminal"):

            if bound_counter == ubound:
                return None          

            current_node["type"] == "nonterminal"
            num_feature = current_node["feature_split"]

            if self._feature_types[num_feature] == "real":
                threshold = current_node["threshold"]
                if x[num_feature] < threshold:
                    current_node = current_node["left_child"]
                else:
                    current_node = current_node["right_child"]

            elif self._feature_types[num_feature] == "categorical":
                splitting_cats = current_node["categories_split"]
                if x[num_feature] in splitting_cats:
                    current_node = current_node["left_child"]
                else:
                    current_node = current_node["right_child"]
                    
            bound_counter += 1
        
        return current_node["class"] 

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
    
    def get_params(self, deep=True):
        return {'feature_types': self._feature_types,
                'max_depth': None,
                'min_samples_split': None,
                'min_samples_leaf': None
               }
