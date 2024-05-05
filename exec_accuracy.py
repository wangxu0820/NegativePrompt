import numpy as np
import random
import utility
import re
import string
from llm_response import get_response_from_llm
import json


def get_query(prompt, eval_template, input_, few_shot, demos_template, demo_data):
    if few_shot == True:
        demos = demos_template.fill(demo_data)
        query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='',
                               full_demo=demos)
    else:
        query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output='')
    # print('DEMOS:', demos)
    return query


def subsample_data(data, subsample_size):
    """
    Subsample data. Data is in the form of a tuple of lists.
    """
    inputs, outputs = data
    assert len(inputs) == len(outputs)
    indices = random.sample(range(len(inputs)), subsample_size)
    inputs = [inputs[i] for i in indices]
    outputs = [outputs[i] for i in indices]
    return inputs, outputs


def exec_accuracy_evaluator(prompts, eval_template, eval_data, llm_model, pnum, task, num_samples, few_shot, demos_template, few_shot_data, num_demos):
    queries = []
    answers = []
    my_inputs = []
    for prompt in prompts:
        subsampled_data = subsample_data(
            eval_data, num_samples)
        inputs, outputs = subsampled_data
        for d in zip(inputs, outputs):
            input_, output_ = d
            demo_data = subsample_data(
                few_shot_data, num_demos)
            query = get_query(prompt, eval_template, input_, few_shot, demos_template, demo_data)
            # query = get_query(
            #     prompt, eval_template, input_)
            queries.append(query)
            answers.append(output_)
            my_inputs.append(input_)

    # get response from LLM
    model_outputs = get_response_from_llm(
        llm_model=llm_model, queries=queries, task=task, few_shot=few_shot)
    # model_outputs = []
    # all_outputs = []
    # with open(f"gpt4_output.json","r") as f:
    #     q_dict = json.load(f)
    #     all_outputs = q_dict['times_2'][task]
    #     model_outputs = all_outputs[pnum * 100: (pnum+1) * 100]

    metric = utility.TASK_TO_METRIC.get(task, utility.default_metric)

    print(f'Using metric "{metric}" for task "{task}"...')

    if metric == 'es':
        score_fn = utility.get_multi_answer_exact_set
    elif metric == 'em':
        score_fn = utility.get_multi_answer_em
    elif metric == 'f1':
        score_fn = utility.get_multi_answer_f1
    elif metric == 'contains':
        score_fn = utility.get_multi_answer_contains

    # postprocess the responses
    if task == 'cause_and_effect':
        new_ans_ = []
        for my_input, ans_ in zip(my_inputs, answers):
            sentences = my_input.split('.')
            for i in range(len(sentences)):
                if ans_[0].lower() in sentences[i].lower() + '.':
                    new_a = f'Sentence {i+1}: ' + ans_[0]
                    new_ans_.append([new_a])
                    break
        answers = new_ans_
    elif task == 'larger_animal':
        new_ans_ = []
        for my_input, ans_ in zip(my_inputs, answers):
            animals = my_input.split(',')
            for i in range(len(animals)):
                if ans_[0].lower() in animals[i].lower():
                    new_a = f'Animal {i}: ' + ans_[0]
                    new_ans_.append([new_a])
                    break
        answers = new_ans_

    # postprocess the responses for different tasks
    for my_input, prediction, ans_ in zip(my_inputs, model_outputs, answers):
        for a in ans_:
            if task == 'cause_and_effect':
                ans_parts = a.split(':')
                for p in ans_parts:
                    p = p.strip().lower()
                    p = p.replace('.', '')
                    if p in prediction.lower():
                        prediction = a
                        break
            elif task == 'rhymes':
                for p in prediction.split():
                    p = p.replace('-', ' ')
                    p = p.translate(
                        str.maketrans('', '', string.punctuation))
                    p = p.strip().lower()
                    if p == a.lower():
                        prediction = a
                        break
            elif task == 'orthography_starts_with':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(
                    str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                for pred in preds:
                    pred = pred.strip()
                preds_set = set(preds)

                a_items = a.split()
                for a in a_items:
                    a = a.strip()
                a_set = set(a_items)

                if a_set == preds_set:
                    prediction = a
            elif task == 'taxonomy_animal':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(
                    str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                for pred in preds:
                    pred = pred.strip()
                preds_set = set(preds)

                a_items = a.split(',')
                for a in a_items:
                    a = a.strip()
                a_set = set(a_items)

                if a_set == preds_set:
                    prediction = a
            elif task == 'letters_list':
                prediction = prediction.lower()
                prediction = prediction.replace('confidence score:', '')
                prediction = prediction.replace(',', ' ')
                prediction = prediction.replace('.', ' ')
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(
                    str.maketrans('', '', string.punctuation))
                prediction = re.sub(r'\d+', '', prediction)
                preds = prediction.split()
                for pred in preds:
                    pred = pred.strip()
                a_items = a.split()
                for a in a_items:
                    a = a.strip()
                if preds == a_items:
                    prediction = a_items

            elif task == 'sentiment':
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(
                    str.maketrans('', '', string.punctuation))
                prediction = prediction.strip().lower()
                if 'does not mention any negative' in prediction or 'a positive review than a negative one' in prediction:
                    prediction = 'positive'
                    break
                elif 'does not mention any positive' in prediction or 'a negative review than a positive one' in prediction:
                    prediction = 'negative'
                    break
                if 'positive' in prediction and 'negative' in prediction:
                    prediction = ''
                    break
                elif 'positive' in prediction or 'positiv' in prediction:
                    prediction = 'positive'
                    break
                elif 'negative' in prediction or 'negativ' in prediction:
                    prediction = 'negative'
                    break
                if len(prediction.split()) == 1:
                    prediction = postprocess_prediction_4sentiment(prediction)
                elif len(prediction.split()) > 1:
                    items = prediction.split()
                    new_res = postprocess_prediction_4sentiment(
                        items[0].strip())
                    if new_res == 'positive' or new_res == 'negative':
                        prediction = new_res
                if a in prediction:
                    prediction = a
                    break
            elif task == 'sentence_similarity':
                a_score = a.split()[0]
                prediction = prediction.replace('-', ' ')
                prediction = prediction.translate(
                    str.maketrans('', '', string.punctuation))
                prediction = prediction.strip().lower()
                prediction_list = prediction.split()
                for item in prediction_list:
                    if item.isdigit():
                        p = item
                        p_score = p[0]
                        if p_score == a_score:
                            prediction = a_score
                        break
            elif task == 'word_in_context':
                prediction = prediction.strip().lower()
                if len(prediction.split()) > 0:
                    p = prediction.split()[0]
                    p = p.replace('-', ' ')
                    p = p.translate(
                        str.maketrans('', '', string.punctuation))
                    p = p.strip()
                    if p == 'true' or p == 'yes' or p == '1' or p == '10' or p == 'same' or 'same' in p or p == 'match' or p == 'similar':
                        prediction = 'same'
                    elif p == 'false' or p == 'no' or p == '0' or p == '00' or p == 'different' or 'different' in p or p == 'not' or p == 'opposite':
                        prediction = 'not the same'
                    elif 'different' in prediction and 'not' not in prediction:
                        prediction = 'not the same'
                    elif 'different' in prediction and 'not' in prediction:
                        prediction = 'same'
                    elif 'same' in prediction and 'not' not in prediction:
                        prediction = 'same'
                    elif 'same' in prediction and 'not' in prediction:
                        prediction = 'not the same'
            elif task == 'larger_animal':
                prediction = prediction.lower()
                if 'larger' in prediction and 'than' in prediction:
                    index = prediction.find('larger')
                    prediction = prediction[:index]
                if 'between' in prediction and 'and' in prediction and 'is' in prediction:
                    index = prediction.find('is')
                    prediction = prediction[index:]
                if 'confidence' in prediction:
                    index = prediction.find('confidence')
                    prediction = prediction[:index]
                    prediction = prediction.strip()
                if llm_model.lower() == 't5' or llm_model.lower() == 'bloom':
                    pred_list = prediction.split()
                    if len(pred_list) > 0:
                        ans_part = pred_list[0]
                        if ',' in ans_part and llm_model.lower() != 'chatgpt':
                            prediction = ''
                if llm_model.lower() == 'bard' and ',' in prediction:
                    pred_list = prediction.split(',')
                    prediction = pred_list[-1]
                a = a.strip().lower()
                a_items = a.split()
                # print(a_items)
                if a in prediction.lower():
                    prediction = a
                    break
                elif len(a_items) > 1:
                    a_2 = a_items[-1].strip()
                    if a_2 in prediction.lower():
                        prediction = a
                        break
                if prediction == '0' and '0' in a:
                    prediction = a
                elif prediction == '1' and '1' in a:
                    prediction = a
                elif '1' in a and ('1.0' in prediction or '1' in prediction or '2' in prediction):
                    prediction = a
                elif '0' in a and ('0.0' in prediction or '0' in prediction):
                    prediction = a
                # a_list = a.split(':')
            else:
                a = a.strip().lower()
                if a in prediction.lower():
                    prediction = a
                    break

        print('Model Input: ', my_input, ' Model Output: ',
              prediction, ' Ans: ', ans_)

    scores = []
    for prediction, ans_ in zip(model_outputs, answers):
        score = score_fn(prediction, ans_, task, llm_model.lower())
        scores.append(score)

    # Reshape the scores so that it is num_prompts x num_samples
    scores = np.array(scores).reshape(len(prompts), num_samples)

    res = ExecAccuracyEvaluationResult(prompts, scores)
    return res


def postprocess_prediction_4sentiment(prediction):
    if prediction == 'neg':
        prediction = 'negative'
    elif prediction == 'pos':
        prediction = 'positive'
    elif prediction.isdigit() or (prediction[0] == '-' and prediction[1:].isdigit()):
        p_digit = int(prediction)
        if p_digit > 0:
            prediction = 'positive'
        else:
            prediction = 'negative'
    return prediction


class ExecAccuracyEvaluationResult():

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        if method == 'mean':
            return [np.mean(s) for s in self.scores]
        elif method == 'median':
            return [np.median(s) for s in self.scores]
        elif method == 'std':
            return [np.std(s) for s in self.scores]
        elif method == 'max':
            return [np.max(s) for s in self.scores]
        elif method == 'min':
            return [np.min(s) for s in self.scores]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.scores]
        else:
            raise ValueError('Invalid method: {}'.format(method))

    def sorted(self):
        scores = [np.mean(s) for s in self.scores]
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores
