## How to run, train, and chart the results of the model


### 0. Install the dependencies
`pip3 install -r requirements.txt`

Install the ROMs for the Atari games, a guide can be seen here:

`https://github.com/openai/atari-py`

### 1. Train the model
The following is an example command which should be able to be run out of the box.

`python3 train.py --render --verbose --epsilon_decay 0.99911 --environment space_invaders --n_step 4 --lr_model 0.001 --frames 4 --frame_skip 4 --batch_size 32 --gamma 0.99 --epsilon_min 0.01 --save --max_episodes 1000 --params_update 4 --memory_capacity 5000 --backup`

If your device supports CUDA, you can use the GPU to train the model. To do so, add the `--use_cuda` flag to the command above.

More information about the flags can be found by running `python3 train.py --help`.

Once the training is completed, or has been stopped via. `Ctrl+C`, the model will be saved inside a new directory, where the metrics will be saved as a `.npy` file.

### 2. Chart the results
Inside the file `make_graphs.py`, change the `file` variable to be the path to the `.npy` metrics file you want to chart. Then, run the file with `python3 make_graphs.py`.

### Common Errors
#### a. RuntimeError: cuDNN error: CUDNN_STATUS_MAPPING_ERROR
This is caused by the frequency of parameters being updated. To fix this, decrement the `--params_update` flags value in the above command.
#### b. RuntimeError: CUDA out of memory.
Same as above.
#### c. Rendering doesn't work, env.render() complains.
This is caused by the `--render` flag. To fix this, remove the `--render` flag from the above command if it is there.
#### d. ValueError: crop_width has an invalid length: 3
This happens if you've installed the ROMs via. ale (`A.L.E. (Arcade Learning Environment)`) instead of `atari-py`. To fix this, uninstall `ale-py` and install `atari-py` instead. (See https://github.com/openai/atari-py)