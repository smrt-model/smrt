""" 
Provides general tools related to the code repository.
"""

# Way to print out mercurial commit number
# See http://stackoverflow.com/questions/1153469/mercurial-scripting-with-python
import subprocess


def get_hg_rev(file_path):
    """
    Gets the mercurial commit ID of the model you are using.

    This is useful when revisiting ipython notebooks, and can be used to compare the original
    model commit ID with the latest version.

    Usage:
        from smrt.utils.repo_tools import get_hg_rev
        path_to_file = "/path/to/your/repository"
        get_hg_rev(path_to_file)

    Note:
        This is for a mercurial repository.

    Args:
        file_path: Path to the repository.

    Returns:
        The commit ID as bytes.
    """
    pipe = subprocess.Popen(
        ["hg", "id", "-i", "-R", file_path],
        stdout=subprocess.PIPE
    )
    return pipe.stdout.read()
