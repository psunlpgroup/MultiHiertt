"""MathQA utils.
"""
import json
from tqdm import tqdm

def str_to_num(text):
    text = text.replace("$","")
    text = text.replace(",", "")
    text = text.replace("-", "")
    text = text.replace("%", "")
    try:
        num = float(text)
    except ValueError:
        if "const_" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            num = float(text)
        else:
            num = "n/a"
    return num


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token) or (str_to_num(num) != "n/a" and str_to_num(num) / 100 == str_to_num(token)):
                        cur_num_idx = num_idx
                        break
                    
            if cur_num_idx == -1:
                return None
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')



def read_txt(input_path):
    """Read a txt file into a list."""
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items

def get_op_const_list():
    op_list_file = "../txt_files/operation_list.txt"
    const_list_file = "../txt_files/constant_list.txt"
    op_list = read_txt(op_list_file)
    op_list = [op + '(' for op in op_list]
    op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
    const_list = read_txt(const_list_file)
    const_list = [const.lower().replace('.', '_') for const in const_list]
    return op_list, const_list


def write_predictions(all_predictions, output_prediction_file):
    """Writes final predictions in json format."""

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")