import fire
from data.instruction_induction.load_data import load_data, tasks
from exec_accuracy import exec_accuracy_evaluator
from config import PROMPT_SET, Negative_SET, APE_PROMPT_SET, APE_PROMPTs
import template
import os
import random

def getPrompt(ori_prompt, num_str):
    new_prompt = ori_prompt
    if num_str > 0:
        new_prompt = ori_prompt + Negative_SET[num_str - 1]
    return new_prompt


def run(task, model, pnum, few_shot):
    assert task in tasks, 'Task not found!'

    test_data = load_data('eval', task)
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\nAnswer: [OUTPUT]"
    # origin_prompt = PROMPT_SET[task]
    origin_prompt = APE_PROMPTs[task]

    # few-shot setting 
    induce_data = load_data('induce', task)
    few_shot_data = induce_data[0], [random.sample(output, 1)[0]
                                        for output in induce_data[1]]
    num_demos = 5
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\n[full_DEMO]\nInput: [INPUT]\nAnswer: [OUTPUT]"
    demos_template = template.DemosTemplate(demos_template)

    # Evaluate on test data
    print('LLM: ', model)
    print('Evaluating on test data...')

    new_prompt = getPrompt(origin_prompt, pnum)
    print('Prompt: ', new_prompt)
    print('Few_shot: ', few_shot)

    test_num = min(100, len(test_data[0]))

    # p_list = APE_PROMPT_SET[task]
    eval_template = template.EvalTemplate(eval_template)
    # for p in p_list:
    test_res = exec_accuracy_evaluator(prompts=[new_prompt],
                                    eval_template=eval_template,
                                    eval_data=test_data,
                                    llm_model=model, pnum=pnum,
                                    task=task,
                                    num_samples=test_num,
                                    few_shot=few_shot,
                                    demos_template = demos_template,
                                    few_shot_data=few_shot_data,
                                    num_demos=num_demos)

    test_score = test_res.sorted()[1][0]

    print(f'Test score: {test_score}')

    if few_shot == True:
        dir_path = f'results/ape_{model}_True'
    else:
        dir_path = f'results/ape_{model}'
    if os.path.exists(dir_path) == False:
        os.makedirs(dir_path)

    with open(f'{dir_path}/{task}.txt', 'a+') as f:
        f.write(f'Test score: {test_score}\n')
        f.write(f'Prompt(few-shot: {few_shot}): {new_prompt}\n')


if __name__ == '__main__':
    fire.Fire(run)
