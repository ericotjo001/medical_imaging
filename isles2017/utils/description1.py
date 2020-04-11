DESCRIPTION = '''\t\t\t=== Welcome to main.py of a medical imaging project! ===

Implementations of 3D versions of Neural Network to handle ISLES 2017 Ischemic Stroke Lesion Segmentation.
  See http://www.isles-challenge.org/ISLES2017/
  Send me email at ericotjoa@gmail.com if you would like to a sample of trained model.
  Currently available:
    1. U-Net + LRP. Baseline performance on training dataset (overfit) can reach the state of the art. Dice score 0.3~0.6 
    2. (__IN_PROGRESS__) U-Net + data augmentation using artifical noise (see dataio/data_diffgen.py)

Minimalistic step-by-step instructions:
  1. run python main.py --mode create_config_file --config_dir config.json 
  2. edit the directories of your ISLES2017 data in the config.json
  3. choose training_mode and evaluation_mode (see below)
  4. run python main.py --mode train --config_dir config.json. The outputs should be in "checkpoints" folder.
  5. run python main.py --mode evaluation --config_dir config.json. The outputs should be in "checkpoints" folder.

Tips: 
  (+) See entry.py shortcut_sequence to create custom training sequences.
  (+) DEBUG modes in utils/debug_switches.py are convenient for debugging. Try them out.

Configuration files:
  Directory to config file:
    > Set --config_dir __path_to_config_dir__ in the modes below if you want to use
      config file from elsewhere. The default directory is config.json
    > Warning: Not all entries in the configuration files are relevant to specific implementations. 
      For example, in one particular instance of our implementation, when learning mechanism is adam, 
      momentum is not used. When SGD is used as the learning mechanism, betas are not relevant.

Modes:
  (1) info
    python main.py
    python main.py --mode info  
    python main.py --mode more_info_on_publication_1
      > The above mode is to provide information on the following paper:
        "Enhancing the Extraction of Interpretable Information 
        for Ischemic Stroke Imaging from Deep Neural Networks."
    python main.py --mode more_info_on_publication_2
      > __TO_BE_IMPLEMENTED__


Basic processes:
  (2) create_config_file
    python main.py --mode create_config_file
    python main.py --mode create_config_file --config_dir config.json 

  (3) test. Ad hoc testing.
    python main.py --mode test

  (4) train. Training network.
    python main.py --mode train

  (5) evaluation. Use trained network to perfrom task, or evaluate a trained network.
    python main.py --mode evaluation

  (6) lrp. Use Layerwise Relevance Propagation (LRP) for interpretability studies. See description above.
    python main.py --mode lrp

  (7) visual.
    python main.py --mode visual

  (X) shorcut_sequence. 
    python main.py --mode shortcut_sequence --shortcut_mode XX1
    python main.py --mode shortcut_sequence --shortcut_mode diff_gen_shortcut
'''