"""GAM baselines adapted from https://github.com/zzzace2000/GAMs_models/."""


import os
import time


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 \
            else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


def output_csv(the_path, data_dict, order=None, delimiter=','):
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    with open(the_path, 'a+') as op:
        keys = list(data_dict.keys())
        if order is not None:
            keys = order + [k for k in keys if k not in order]

        col_title = delimiter.join([str(k) for k in keys])
        if not is_file_exists:
            print(col_title, file=op)
        else:
            old_col_title = open(the_path, 'r').readline().strip()
            if col_title != old_col_title:
                old_order = old_col_title.split(delimiter)

                additional_keys = [k for k in keys if k not in old_order]
                if len(additional_keys) > 0:
                    print('WARNING! The data_dict has following additional keys %s'
                          % (str(additional_keys)))

                no_key = [k for k in old_order if k not in keys]
                if len(no_key) > 0:
                    raise(RuntimeError('The data_dict does not have the following old keys: %s'
                                       % str(no_key)))

                keys = old_order + additional_keys

        print(delimiter.join([str(data_dict[k]) for k in keys]), file=op)


def vector_in(vec, names):
    is_kept = (vec == names[0])
    for m_name in names[1:]:
        is_kept = (is_kept | (vec == m_name))
    return is_kept
