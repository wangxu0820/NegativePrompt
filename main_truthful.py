from data.TruthfulQA.load_data import load_data
from fine_tuned_gpt_eval import fine_tuned_gpt_evaluator
from config import EMOTION_SET
import fire
import template
import os


def getStimulus(num_str):
    stimulus = ''
    if num_str > 0:
        stimulus += EMOTION_SET[num_str - 1]
    return stimulus


def run(pnum, model, api_num):
    test_data = load_data('all')
    eval_template = "Q: [INPUT]\nA: [OUTPUT]"
    eval_template = template.EvalTemplate(eval_template)
    # questions, answers = test_data
    cur_stimulus = getStimulus(pnum)
    # new_prompts = getPrompt(questions, pnum)
    print('LLM: ', model)
    print('pnum: ', pnum)
    truth_score, info_score = fine_tuned_gpt_evaluator(cur_stimulus, eval_template, test_data, model, api_num)

    print(f'Truth score: {truth_score}')
    print(f'Info score: {info_score}')

    dir_path_1 = f'results/truthful'
    if os.path.exists(dir_path_1) == False:
        os.mkdir(dir_path_1)
    dir_path_2 = f'results/information'
    if os.path.exists(dir_path_2) == False:
        os.mkdir(dir_path_2)

    with open(f'results/truthful/{model}.txt', 'a+') as f:
        f.write(f'Truth score: {truth_score}\n')
        f.write(f'Stimulus {pnum}: {cur_stimulus}\n')

    with open(f'results/information/{model}.txt', 'a+') as f:
        f.write(f'Info score: {info_score}\n')
        f.write(f'Stimulus {pnum}: {cur_stimulus}\n')


if __name__ == '__main__':
    fire.Fire(run)

