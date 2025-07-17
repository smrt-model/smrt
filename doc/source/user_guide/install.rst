####################################
Install SMRT
####################################

To get started with SMRT, you will need to install the latest stable version.

### Option 1: Install the latest stable SMRT package

Using `pip` or `conda` in an existing virtual environnement.

1.  **Install directly from the repository**:
    ```bash
    pip install smrt
    ```

If you need features that are still under development, you may want to install the latest developers' version of SMRT

### Option 2: Using `pip` for Editable Installation

Using `pip` and `venv` (or `conda`), you can install the project in editable mode.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/smrt-model/smrt.git
    cd smrt
    ```

2.  **Create and activate a virtual environment**
    
    This can be done for example with `venv` but please refer to https://docs.python.org/3/library/venv.html if this is new to you. Most IDE have their own way of generating virtual environments, which may be easier than using `venv`.

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```
    

3.  **Install the project in editable mode**:
    ```bash
    pip install '.'
    ```
