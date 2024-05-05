import re
import string
from collections import Counter

TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'em',
                  'taxonomy_animal': 'es', 'synonyms': 'contains'}
default_metric = 'em'


def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.replace('don\'t', 'do not')
    prediction = prediction.replace('can\'t', 'cannot')
    prediction = prediction.replace('doesn\'t', 'does not')
    prediction = prediction.replace('wasn\'t', 'was not')
    prediction = prediction.replace('weren\'t', 'were not')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    # print('P: ', prediction_normalized)
    return prediction_normalized == ground_truth_normalized


def get_em_score_contain(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if ground_truth_normalized in prediction_normalized:
        return 1
    return 0


def get_em_score_sentiment(prediction, ground_truth):
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))
    prediction = prediction.strip().lower()
    if 'positive' in prediction and 'negative' in prediction:
        return 0
    elif 'positive' in prediction:
        prediction = 'positive'
    elif 'negative' in prediction:
        prediction = 'negative'
    if len(prediction.split()) == 1:
        prediction = postprocess_prediction_4sentiment(prediction)
    elif len(prediction.split()) > 1:
        items = prediction.split()
        new_res = postprocess_prediction_4sentiment(items[0].strip())
        if new_res == 'positive' or new_res == 'negative':
            prediction = new_res
        elif 'positive' in prediction or 'positiv' in prediction:
            prediction = 'positive'
        elif 'negative' in prediction or 'negativ' in prediction:
            prediction = 'negative'
    if ground_truth in prediction:
        return 1
    return 0


def get_em_score_cause_effect(prediction, ground_truth):
    ans_parts = ground_truth.split(':')
    for a in ans_parts:
        a = a.strip().lower()
        a = a.replace('.', '')
        if a in prediction.lower():
            return 1
    return 0


def get_em_score_rhymes(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    for word in prediction_normalized.split():
        word = word.lower().strip()
        if word == ground_truth_normalized:
            return 1
    return 0


def get_em_score_starts_with(prediction, ground_truth):

    prediction = prediction.lower()
    prediction = prediction.replace('confidence score:', '')
    prediction = prediction.replace('score:', '')
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

    a_items = ground_truth.split()
    for a in a_items:
        a = a.strip()
    a_set = set(a_items)

    if a_set == preds_set:
        return 1

    print('Wrong: ', 'gt:', ground_truth, ' predict: ', prediction)
    return 0


def get_em_score_letters_list(prediction, ground_truth):

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
    a_items = ground_truth.split()
    for a in a_items:
        a = a.strip()
    if preds == a_items:
        return 1
    
    return 0


def get_em_score_taxonomy_animal(prediction, ground_truth):
    
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

    a_items = ground_truth.split(',')
    for a in a_items:
        a = a.strip()
    a_set = set(a_items)

    if a_set == preds_set:
        return 1
    print('Wrong! ', ground_truth, prediction)
    
    return 0


def get_em_score_sentence_similarity(prediction, ground_truth):
    a_score = ground_truth.split()[0]
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
                return 1
    return 0


def get_em_score_word_in_context(prediction, ground_truth):
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
    if prediction == ground_truth:
        return 1
    return 0


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


def get_em_score_larger_animal(prediction, ground_truth, model):
    prediction = prediction.lower()
    if 'larger' in prediction and 'than' in prediction:
        index = prediction.find('larger')
        prediction = prediction[:index]
    if 'confidence' in prediction:
        index = prediction.find('confidence')
        prediction = prediction[:index]
        prediction = prediction.strip()
    if model.lower() == 't5' or model.lower() == 'bloom':
        # 这个针对T5（以及Bloom）
        pred_list = prediction.split()
        if len(pred_list) > 0:
            ans_part = pred_list[0]
            if ',' in ans_part and model.lower() != 'chatgpt':
                return 0
    if model.lower() == 'bard' and model.lower() == 'vicuna':
        # 这个针对Bard
        if ',' in prediction:
            pred_list = prediction.split(',')
            prediction = pred_list[-1]
    ground_truth = ground_truth.strip().lower()
    a_items = ground_truth.split()
    ground_truth = ground_truth.strip().lower()
    if ground_truth in prediction.lower():
        return 1
    elif len(a_items) > 1:
        a_2 = a_items[-1].strip()
        if a_2 in prediction.lower():
            return 1
    if prediction == '0' and '0' in ground_truth:
        return 1
    elif prediction == '1' and '1' in ground_truth:
        return 1
    elif '1' in ground_truth and ('1.0' in prediction or '1' in prediction or '2' in prediction):
        return 1
    elif '0' in ground_truth and ('0.0' in prediction or '0' in prediction):
        return 1
    return 0


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    if int(set(prediction_normalized) == set(ground_truth_normalized)) == 1:
        return 1
    else:
        ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split(',')
        if int(set(prediction_normalized) == set(ground_truth_normalized)) == 1:
            return 1
        else:
            print('Wrong: ', 'gt:', ground_truth, ' predict: ', prediction)
            return 0
        # return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1


def get_multi_answer_em(prediction, answers, task, model):
    for answer in answers:
        if task.lower() == 'sentiment':
            if get_em_score_sentiment(prediction, answer) == 1:
                return 1
        elif task.lower() == 'sentence_similarity':
            if get_em_score_sentence_similarity(prediction, answer) == 1:
                return 1
        elif task.lower() == 'larger_animal':
            if get_em_score_larger_animal(prediction, answer, model) == 1:
                return 1
        elif task.lower() == 'sum' or task.lower() == 'diff' or task.lower() == 'antonyms' or task.lower() == 'singular_to_plural' or task.lower() == 'translation_en-de' or task.lower() == 'translation_en-es' or task.lower() == 'translation_en-fr':
            if get_em_score_contain(prediction, answer) == 1:
                return 1
            # if get_em_score(prediction, answer) == 1:
            #     return 1
        elif task.lower() == 'orthography_starts_with':
            if get_em_score_starts_with(prediction, answer) == 1:
                return 1
        elif task.lower() == 'taxonomy_animal':
            if get_em_score_taxonomy_animal(prediction, answer) == 1:
                return 1
        elif task.lower() == 'letters_list':
            if get_em_score_letters_list(prediction, answer) == 1:
                return 1
        elif task.lower() == 'word_in_context':
            if get_em_score_word_in_context(prediction, answer) == 1:
                return 1
        elif task.lower() == 'cause_and_effect':
            if get_em_score_cause_effect(prediction, answer) == 1:
                return 1
        elif task.lower() == 'rhymes':
            if get_em_score_rhymes(prediction, answer) == 1:
                return 1
        elif task.lower() == 'first_word_letter' or task.lower() == 'second_word_letter':
            if 'is' in prediction:
                index = prediction.find('is')
                if index + 2 < len(prediction):
                    prediction = prediction[index+2:]
                    if get_em_score_contain(prediction, answer) == 1:
                        return 1
            elif 'would be' in prediction:
                index = prediction.find('would be')
                if index + 8 < len(prediction):
                    prediction = prediction[index+8:]
                    if get_em_score_contain(prediction, answer) == 1:
                        return 1
            else:
                if get_em_score(prediction, answer) == 1:
                    return 1
            print('Wrong: ', answer, prediction)
        else:
            if get_em_score(prediction, answer) == 1:
                return 1

    return 0


def get_multi_answer_exact_set(prediction, answers, task, modelc):
    for answer in answers:
        if get_exact_set_score(prediction, answer) == 1:
            return 1
        # print('Wrong: ', answer, prediction)
    return 0


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_multi_answer_f1(prediction, answers, task, model):
    f1_scores = []
    for answer in answers:
        f1_scores.append(get_f1_score(prediction, answer))
    return max(f1_scores)

def get_multi_answer_contains(prediction, answers, task, model):
    for answer in answers:
        if get_contains_score(prediction, answer) == 1:
            return 1
    return 0
