import itertools
import os.path
import pdb
import copy

import causal_nf.utils.io as causal_io


def get_value(value):
    if isinstance(value, str):
        value = eval(value)
    return value


def generate_options_v2(grid_flat: dict, correlations: str = 'correlations', steps: str = 'steps_correlations', grid_file_extra: dict=None) -> list:
    """
        We already are relying on the correct ordering of values in options when we send them further for 
        processing in generate_jobs.py. So we'll use this to infer which is the correlation value and step_size
        so we can modify them, with minimal changes to the rest of the code.

        Remark: be mindful, because this will explode the number of jobs.
    """
    
    values = []
    grid_flat_extra = None
    if isinstance(grid_file_extra, str) and os.path.exists(grid_file_extra):
        grid_flat_extra = causal_io.load_yaml(grid_file_extra, flatten=True)

    for idx, elem in enumerate(grid_flat.keys()):
        if correlations in elem:
            corr_idx = idx
        if steps in elem:
            steps_idx = idx
    # pdb.set_trace()
    for key, value in grid_flat.items():
        if isinstance(value, str):
            assert value == "TODO", f"value: {value}"
            value = grid_flat_extra[key]

        value = get_value(value)
        assert isinstance(value, list), f"key | value: {key} | {value}"
        if key == 'dataset__steps':
            # pdb.set_trace()
            pass
        values.append(value)
    # pdb.set_trace()
    options = list(itertools.product(*values))
    """
        Every option in options will now have option[corr_idx] a list of correlations, and at option[steps_idx] how many staps to take for it
    """
    output = []
    for i in range(len(options)):
        _steps = options[i][steps_idx]
        if _steps != 0:
            for s in range(_steps + 1):
                _option = copy.deepcopy(options[i]) 
                for j, elem in enumerate(options[i][corr_idx]):
                    """
                        Individual correlations are here. Let's assume it's just normal pdfs
                    """
                    _option[corr_idx][j][-1] = round(options[i][corr_idx][j][-1] * s / _steps, 4)
                
                output.append(_option)
        else:
            _option = copy.deepcopy(options[i]) 
            output.append(_option)
    # pdb.set_trace()
    return output

def generate_options(grid_flat, grid_file_extra=None):
    values = []
    grid_flat_extra = None
    if isinstance(grid_file_extra, str) and os.path.exists(grid_file_extra):
        grid_flat_extra = causal_io.load_yaml(grid_file_extra, flatten=True)
    for key, value in grid_flat.items():
        if isinstance(value, str):
            assert value == "TODO", f"value: {value}"
            value = grid_flat_extra[key]
        value = get_value(value)
        assert isinstance(value, list), f"key | value: {key} | {value}"
        values.append(value)
    options = list(itertools.product(*values))
    return options


def get_grid_file_extra_list(grid_file):
    folder = os.path.dirname(grid_file)
    grid_basename = os.path.basename(grid_file)
    grid_name = os.path.splitext(grid_basename)[0]
    grid_list = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        cond1 = grid_basename != file
        cond2 = grid_name in file
        cond2 = grid_name == file[: len(grid_name)]
        cond3 = os.path.isfile(file_path)
        if cond1 and cond2 and cond3:
            grid_list.append(file_path)
    return grid_list
