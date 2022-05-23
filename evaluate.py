import json, os, re
from utils.span_selection_utils import *
from utils.program_generation_utils import *
import math 

def evaluate_program_result(pred_prog, gold_prog):
    '''
    execution acc
    execution acc = exact match = f1
    '''
    invalid_flag, exe_res = eval_program(pred_prog)

    gold = program_tokenization(gold_prog)
    invalid_flag, exe_gold_res = eval_program(gold)

    if invalid_flag:
        print(gold)
    if exe_res == exe_gold_res:
        exe_acc = 1
    else:
        exe_acc = 0

    return exe_acc, exe_acc

def evaluate_span_program_result(span_ans, prog_ans):
    span_ans = str(span_ans)
    if str_to_num(span_ans) != "n/a":
        span_ans = str_to_num(span_ans)
        if math.isclose(prog_ans, span_ans, abs_tol= min(abs(min(prog_ans, span_ans) / 1000), 0.1)):
            exact_match, f1 = 1, 1
        else:
            exact_match, f1 = 0, 0
    else:
        exact_match, f1 = get_span_selection_metrics(span_ans, str(prog_ans))
    return exact_match, f1

def combine_predictions(span_selection_json_in, program_generation_json_in, test_file_json_in, output_dir):
    span_selection_data = json.load(open(span_selection_json_in))
    program_generation_data = json.load(open(program_generation_json_in))
    orig_data = json.load(open(test_file_json_in))

    prediction_dict = {}
    for example in span_selection_data + program_generation_data:
        uid = example["uid"]
        pred_ans = example["predicted_ans"]
        pred_program = example["predicted_program"]

        if uid in prediction_dict:
            print(f"uid {uid} already in prediction_dict")
        else:
            prediction_dict[uid] = {
                "uid": uid,
                "predicted_ans": pred_ans,
                "predicted_program": pred_program
            }
    
    output_data = []
    for example in orig_data:
        output_data.append(prediction_dict[example["uid"]])

    mode = "dev" if "dev" in test_file_json_in else "test"
    output_file = os.path.join(output_dir, f"{mode}_predictions.json")
    json.dump(output_data, open(output_file, "w"), indent=4)

    print(f"{mode}: Combine {len(span_selection_data)} examples from span selection output, {len(program_generation_data)} examples from program generation output. The prediction are generated in {output_file}")

    return prediction_dict


def evaluation_prediction_result(span_selection_json_in, program_generation_json_in, test_file_json_in, output_dir):
    exact_match_total, f1_total = 0, 0
    prediction_dict = combine_predictions(span_selection_json_in, program_generation_json_in, test_file_json_in, output_dir)

    if "test" in test_file_json_in:
        print("Please submit the test prediction file to CodaLab to get the results")
        return 

    orig_data = json.load(open(test_file_json_in))
    num_examples = len(orig_data)

    for example in orig_data:
        uid = example["uid"]
        pred = prediction_dict[uid]

        gold_prog = example["qa"]["program"]
        gold_ans = example["qa"]["answer"]

        # both program generation
        if pred["predicted_program"] and gold_prog:
            exact_acc, f1_acc = evaluate_program_result(pred["predicted_program"], gold_prog)
        # both span selection
        elif not pred["predicted_program"] and not gold_prog:
            exact_acc, f1_acc = get_span_selection_metrics(pred["predicted_ans"], gold_ans)
        # gold is span selection, pred is program generation
        elif not pred["predicted_program"] and gold_prog:
            exact_acc, f1_acc = evaluate_span_program_result(span_ans = pred["predicted_ans"], prog_ans = gold_ans)
        # gold is program generation, pred is span selection
        elif pred["predicted_program"] and not gold_prog:
            exact_acc, f1_acc = evaluate_span_program_result(span_ans = gold_ans, prog_ans = pred["predicted_ans"])

        exact_match_total += exact_acc
        f1_total += f1_acc
    exact_match_score, f1_score = exact_match_total / num_examples, f1_total / num_examples
    print(f"Exact Match Score: {exact_match_score}, F1 Score: {f1_score}")

    return exact_match_score, f1_score



if __name__ == '__main__':
    test_path = sys.argv[1]
    if "dev" in test_path:
        mode = "dev"
    elif "test" in test_path:
        mode = "test"
    else:
        raise ValueError("Cannot recognize the file name")

    output_dir = "output"
    span_selection_dir = "span_selection_output"
    program_generation_dir = "program_generation_output"

    span_selection_json_in = os.path.join(output_dir, span_selection_dir, f"{mode}_predictions.json")
    program_generation_json_in = os.path.join(output_dir, program_generation_dir, f"{mode}_predictions.json")
    test_file_json_in = os.path.join("dataset", f"{mode}.json")

    prediction_output_dir = os.path.join(output_dir, "final_predictions")
    os.makedirs(prediction_output_dir, exist_ok=True)
    evaluation_prediction_result(span_selection_json_in, program_generation_json_in, test_file_json_in, prediction_output_dir)
