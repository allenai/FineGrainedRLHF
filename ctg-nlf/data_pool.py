from typing import List
from copy import deepcopy
from collections import defaultdict


class DataPool:
    # DOCUMENTED
    def __init__(self, feedback_types, num_quantiles, num_attributes):
        """
        Initialize a data pool for organizing and managing data into quantiles.

        Args:
            feedback_types (List[List[str]]): A list of possible feedback/reward tokens associated with each quantile,
            for each attribute we care of, e.g., Relevancy, Factuality, and Completeness.
            num_quantiles (int): The number of quantiles to divide the data pool into.

        Note:
            The `feedback_types` list should contain feedback associated with each quantile (len(feedback_types) == num_quantiles).

            Quark-like:
                num_quantiles = 5
                num_attributes = 3
                feedback_types = [
                    [_TREE_TOKEN_00000, _TREE_TOKEN_00001, _TREE_TOKEN_00002, _TREE_TOKEN_00003, _TREE_TOKEN_00004],
                    [_TREE_TOKEN_00005, _TREE_TOKEN_00006, _TREE_TOKEN_00007, _TREE_TOKEN_00008, _TREE_TOKEN_00009],
                    [_TREE_TOKEN_00010, _TREE_TOKEN_00011, _TREE_TOKEN_00012, _TREE_TOKEN_00013, _TREE_TOKEN_00014]
                ]
            
            NLF:
                num_quantiles = 5
                num_attributes = 3
                feedback_types = [
                    [Very relevant, Majorly relevant, Moderately relevant, Slightly irrelevant, Completely irrelevant],
                    [Very factual, Majorly factual, Moderately factual, Slightly factual, Completely factual],
                    [Very complete, Majorly complete, Moderately complete, Slightly complete, Completely complete]
                ]
        """
        self.feedback_types = feedback_types
        self.num_quantiles = num_quantiles
        self.num_attributes = num_attributes

        self.score_pool = defaultdict(list)
        for i in range(self.num_attributes):
            self.score_pool[f"attr_{str(i)}"]
        self.prompt_pool, self.response_pool, self.feedback_pool = [], [], []

    # DOCUMENTED
    def add(self, prompts: List[str], responses: List[str], scores: List[List[float]]):
        """
        Add data to the data pool and organize it into quantiles.

        Args:
            prompts (List[str]): A list of input prompts.
            responses (List[str]): A list of response sequences.
            scores (List[List[float]]): A list of reward scores (i.e., Relevancy, Factuality, and Completeness) 
            corresponding to the responses for every attribute.

        Note:
            - Data is sorted by reward scores, from highest to lowest reward, and feedback/reward tokens are assigned to samples based on quantile ranking.
            - Quantile 0 is associated with highest reward (e.g., highest relevancy), and Quantile 4 is associated with lowest reward (e.g., lowest relevancy)!
        """
        self.prompt_pool.extend(prompts)
        self.response_pool.extend(responses)
        for i, scores_list in enumerate(scores):
            self.score_pool[f"attr_{str(i)}"].extend(scores_list)

        for attr_type in range(self.num_attributes):
            data = zip(self.prompt_pool, self.response_pool, self.score_pool[f"attr_{str(attr_type)}"])
            data = [x for x in data if x[-1] is not None]
            sorted_data = sorted(data, key=lambda x: x[-1], reverse=True) # sorted from maximum to minimum reward scores
            self.prompt_pool, self.response_pool, self.score_pool = [list(x) for x in list(zip(*sorted_data))]

            # divide data pool into quantiles of roughly equal size (last quantile will be larger if the length of the data is not 
            # divisible by the desired number of quantiles), and obtain the associated quantile index to each sample in the data pool
            quantile_idx = [[i] * (len(sorted_data) // self.num_quantiles) for i in range(self.num_quantiles)]
            quantile_idx = [y for x in quantile_idx for y in x] # unfold list of lists into a single list
            quantile_idx = quantile_idx + [self.num_quantiles - 1] * (len(sorted_data) - len(quantile_idx)) # append indices for the last quantile
            # e.g., quantile_idx will be [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 4] if currently the data pool has length 14 and we want to use 5 quantiles (the last four '4's are added as 14 % 5 != 0)
            
            self.feedback_pool = [self.feedback_types[i] for i in quantile_idx] 
            # feedback_pool will be a list of lists, where each element is the feedback associated to a quantile, e..g, ['Very', 'ĠPositive']

    # DOCUMENTED
    def get_data(self):
        """
        Get the data from the data pool.

        Returns:
            Tuple[List[str], List[str], List[str]: A tuple containing the input prompts, response sequences,
            and feedback associated with quantiles.

        """
        return deepcopy(self.prompt_pool), deepcopy(self.response_pool), deepcopy(self.feedback_pool)

