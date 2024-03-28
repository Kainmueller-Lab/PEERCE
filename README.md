# **APEDIA**: **A**ngiosarcoma **P**D-L1 **E**xpression **DIA**gnostics with Deep Learning

**APEDIA**, **A**ngiosarcoma **P**D-L1 **E**xpression **DIA**gnostics with Deep Learning, aims to enhance the decision quality of pathologists in the assessment of PD-L1 expression in angiosarcoma. Given a whole-slide image, it calculates the Tumor Proportion Score, which indicates the percentate of tumor cells which are PD-L1 positive.

**Note**: This repository is currently under construction. More features and documentation will be added as the project progresses.

## Functionality in the Repository

The APEDIA project includes a series of scripts designed to preprocess data, train models, and predict scores relevant to PD-L1 expression in angiosarcoma. Below is a brief overview of each script and its purpose.

### Training Functionality

1. **Tumor Patch Detector Training**
   - **Script Name**: `train_tumor_patch_detector.py`
   - **Description**: Trains a model to detect tumor areas in patches. This script can be executed from the terminal. Upon completion, it outputs a dictionary (in pickle format) containing detailed training logs, such as loss and accuracy metrics over epochs. Additionally, the trained model's weights are saved in the specified output directory, facilitating further analysis, evaluation, or deployment.

2. **Cell Type Detector Training**
   - **Script Name**: `train_cell_type_detector.py`
   - **Description**: Trains a model to differentiate between PD-L1 positive tumor cells, PD-L1 negative tumor cells, and other cells within tumor patches.

### Preprocessing Funtionality

- **Preprocess Data for Cell Type Detection Training**
  - **Script Name**: `preprocess_cell_type_data.py`
  - **Description**: Leverages trained generalist models to transform sparse cell type annotations into annotations suitable for training. *Cellpose* is used to identify cell instances and outline the nuclei, eliminating the need for manual annotation of each cell's outline. A *DeepLIIF* generative adverserial network is employed to isolate the hema components from heavily PD-L1 stained areas, thereby enhancing Cellpose's precision in identifying cell instances. This preprocessing step is required for preparing the dataset for accurate cell type detection training.

### Prediction Functionality

- **Predict Tumor Proportion Score (TPS) for WSIs**
  - **Script Name**: `predict_tps_wsi.py`
  - **Description**: Predicts the Tumor Proportion Score for a whole-slide image, integrating outputs from both the tumor patch and cell type detection model to provide a comprehensive PD-L1 expression score. 

## Getting Started

To set up APEDIA for development or usage, follow these steps:

1. **Clone this Repository**

    ```sh
    git clone https://github.com/Kainmueller-Lab/APEDIA
    cd APEDIA
    ```

2. **Create a new Python environment (Recommended)**  
It is recommended to create a new environment using Python 3.8 (tested) or higher to avoid any compatibility issues with the dependencies. You can create a new environment using Conda or venv. For Conda, you can install the environment directly from the provided `environment.yml` file:

    Using Conda: 

      ```sh
      conda env create -f environment.yml
      conda activate apedia_env
      ```  
      If you prefer using pip to manage packages, you can install the dependencies from the `requirements.txt` file:
      ```sh
      pip install -r requirements.txt
      ```

3. **Install the package**

    ```
    pip install -e .
    ```

## How to Use

The functionality provided by APEDIA can be accessed in several ways, catering to different use cases and preferences.  
Below are the methods to run the function to predicting the Tumor Proportion Score (TPS) from whole-slide images (WSIs).

1. **Script Execution**  
   The scripts included in the APEDIA project, like `predict_tps_wsi.py`, can be run directly from the command line. This method is straightforward and allows for quick execution of tasks. For example:

    ```sh
    python predict_tps_wsi.py --ometiff_path path/to/example.ome.tiff --output_folder path/to/output/folder/
    ```

2. **CLI Command**  
   APEDIA provides a Command Line Interface (CLI) to facilitate easy access to its functionalities. The CLI commands abstract the script executions and offer a user-friendly way to interact with APEDIA. To predict the TPS using the CLI:

    ```sh
    apedia predict_tps_wsi --ometiff_path path/to/example.ome.tiff --output_folder path/to/output/folder/
    ```

3. **Jupyter Notebook Tutorial**  
   For those who prefer an interactive approach or wish to understand the process in greater detail, the `predict_tps_wsi.ipynb` notebook offers a comprehensive guide. This notebook not only allows you to run the TPS prediction but also provides further information and context about the process. You can find this notebook in the `jupyter_notebook_tutorials` folder within the repository.

4. **Importing as a Module**  
   If you are developing a Python project and wish to integrate APEDIA's functionality directly, you can import the required functions into your Python scripts. For instance:

    ```python
    from apedia.predict_tps_wsi import predict_tps_wsi

    # Usage example
    predict_tps_wsi(ometiff_path='path/to/example.ome.tiff', output_folder='path/to/output/folder/')
    ```


## License

APEDIA is open-sourced under the MIT License. This license enables anyone to freely use, modify, and distribute the project, subject to the conditions outlined in the LICENSE file found in the repository.
