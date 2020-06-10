from config import *
from utils import *
from vis import *

if __name__ == '__main__':
    config = get_config()  # get configuration parameters

    _, _, test_set, (char_to_int, int_to_char) = create_datasets(config)

    # visualize cell state values
    vis_cell(test_set, int_to_char, char_to_int, config)

    # visualize three gates: input, forget, output
    #vis_gate(test_set, (int_to_char, char_to_int), config)
