description_on_publication_1="""Welcome to a explainable AI + medical imaging project!

This mode is to display more information regarding the paper:
  "Enhancing the Extraction of Interpretable Information for Ischemic Stroke Imaging from Deep Neural Networks."


Overview:
  > Train 3D U-Net on ISLES2017 training dataset for the prediction of lesion position. 
  > Use Layerwise Relevance Propagation (LRP)** to observe which pixels in the MRI scans
    contribute to the prediction of lesion position. 
  > 3D U-Net are not optimally trained, LRP output is not optimal too. 
    Hence, we apply an LRP modification and investigate the results.
  ** Layerwise Relevance Propagation (LRP). See this website: http://www.heatmapping.org/

The modes listed here show the streamlined sequence of function calls to get the results
  published in the papers. It is a mini-example version that will lead you to the figures,
  but not replicate the actual results (which were run using shortcut_sequence mode)

python main.py --mode results_publication_1
python main.py --mode results_publication_1 --submode example_process_get_main_figure
  > This process is aimed to obtain main figures showing brain slices, neural network
    predictions and LRP outputs. See figure 1 and 4(A-D) of the paper.
python main.py --mode results_publication_1 --submode example_process_get_filters_comparison_figure
  > This process is aimed to obtain figures comparing filter-processed LRP statistics. 
    See figure 4(E) of the paper.    
"""

description_on_publication_2="""Welcome to a explainable AI + medical imaging project!

This mode is to display more information regarding the paper:
  "Enhancing the Extraction of Interpretable Information for Ischemic Stroke Imaging from Deep Neural Networks."


Overview:
  > _NOT_YET_

The modes listed here show the streamlined sequence of function calls to get the results
  published in the papers. It is a mini-example version that will lead you to the figures,
  but not replicate the actual results.

python main.py --mode results_publication_2
"""