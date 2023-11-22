import torch
import json
from torch.utils.data import Dataset
from data_pool import DataPool

class PromptDataset(Dataset):
    """
    PyTorch Dataset for handling prompts.

    This dataset is designed to work with prompt texts. It encapsulates the text of prompts for language generation tasks.

    Args:
        path (str): The path to a file containing prompt data in a specific format (e.g., JSON).
    """
    def __init__(self, path):
        with open(path, 'r') as file:
            data = json.load(file)

        self.prompts = [item["text"].strip() for item in data]

    def __len__(self):
        """
        Get the total number of prompts in the dataset.

        Returns:
            int: The number of prompts in the dataset.
        """
        return len(self.prompts)

    def __getitem__(self, idx):
        """
        Get a prompt at the specified index.

        Args:
            idx (int): The index of the prompt to retrieve.

        Returns:
            dict: A dictionary containing the prompt text.
        """
        return {'prompt': self.prompts[idx]}


class PromptCollator(object):
    def __init__(self, tokenizer):
        """
        Initialize the PromptCollator with a tokenizer.

        Args:
            tokenizer: The tokenizer used to process the input prompts.
        """
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        """
        Collate prompts for language model input, including tokenization and padding.

        Args:
            sequences (List[dict]): A list of sequences, each represented as a dictionary with a 'prompt' key containing the prompt text.

        Returns:
            torch.Tensor: Padded and tokenized input IDs for the prompts.
            torch.Tensor: Prompt input attention mask.

        Note:
            - Sequences are padded with the tokenizer's pad_token_id.
            - Attention masks are generated to indicate which tokens to attend to and which are padding.
        """
        prompts = [sequence['prompt'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask


class SequenceWithFeedbackDataset(Dataset):
    """
    PyTorch Dataset for handling sequences with feedback.

    This dataset is designed to work with sequences that have feedback for conditioning. It encapsulates query, response,
    and associated feedback data.

    Args:
        data_pool (DataPool): An instance of the DataPool class containing the organized data.
    """
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.feedback = data_pool.get_data()

    def __len__(self):
        """
        Get the total number of sequences in the dataset.

        Returns:
            int: The number of sequences in the dataset.
        """
        return len(self.queries)

    def __getitem__(self, idx):
        """
        Get a sequence at the specified index.

        Args:
            idx (int): The index of the sequence to retrieve.

        Returns:
            dict: A dictionary containing the query, response, and associated feedback.
        """
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'feedback': self.feedback[idx]
                }


class SequenceWithFeedbackCollator(object):
    def __init__(self, tokenizer):
        """
        Initialize the SequenceWithFeedbackCollator with a tokenizer.

        Args:
            tokenizer: The tokenizer used to process the input sequences.
        """
        self.tokenizer = tokenizer
        self.max_source_length = tokenizer.max_input_len
        self.max_target_length = tokenizer.max_generated_len

    def __call__(self, sequences):
        """
        Collate sequences for language model input, including feedback, padding, and attention masking.

        Args:
            sequences (List[dict]): A list of sequences, each represented as a dictionary with keys 'query', 'response', and 'feedback'.

        Returns:
            torch.Tensor: Padded and tokenized query input IDs with feedback prepended (tags "feedback: ", and "input: " added).
            torch.Tensor: Query input attention mask with feedback accounted for.
            torch.Tensor: Padded and tokenized response input IDs.
            torch.Tensor: Response input attention mask.
        """
        queries = [self.tokenizer.feedback_prefix + sequence['feedback'] + " " + self.tokenizer.prompt_prefix + sequence['query'] for sequence in sequences]
        responses = [sequence['response'] for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, 
                                              max_length=self.max_source_length, 
                                              return_tensors="pt", padding=True, truncation=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        response_encodings_dict = self.tokenizer(responses, 
                                                 max_length=self.max_target_length, 
                                                 return_tensors="pt", padding=True, truncation=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask