
import os
import glob
import pickle
from uuid import uuid4
from .error import SMRTError


def honour_all_promises(directory_or_filename, save_result_to=None, show_progress=True):
    """Honour many promises and save the results

    :param directory_or_filename: can be a directory, a filename or a list of them
"""
    if isinstance(directory_or_filename, str):
        directory_or_filename = [directory_or_filename]

    filename_list =[]
    for item in directory_or_filename:
        if os.path.isdir(item):
            filename_list += glob.glob(os.path.join(item, "smrt-promise-*.P"))
        elif os.path.isfile(item):
            filename_list.append(item)
        else:
            raise Runtime("directory_or_filename argument must be an existing directory or a filename or a list of them.")

    for filename in filename_list:
        if show_progress:
            print(filename)
        honour_promise(filename, save_result_to or os.path.dirname(filename))
    if show_progress:
        print("done!")

def honour_promise(filename, save_result_to=None):
    """Honour a promise and optionally save the result"""

    promise = load_promise(filename)
    result = promise.run()
    if save_result_to is not None:
        if os.path.isdir(save_result_to):
            outfilename = os.path.join(save_result_to, promise.result_filename)
        elif os.path.isfile(save_result_to):
            outfilename = save_result_to
        else:
            raise Runtime("save_result_to argument must be a directory or a filename")
        result.save(outfilename)


def load_promise(filename):

    with open(filename, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, RunPromise):
        raise SMRTError("The file does not contain a SMRT promise")

    return obj


class RunPromise(object):

    def __init__(self, model, sensor, snowpack, kwargs):
        self.model = model
        self.sensor = sensor
        self.snowpack = snowpack
        self.kwargs = kwargs  # options and other optional arguments
        self.result_filename = None

    def run(self):

        return self.model.run(self.sensor, self.snowpack, **self.kwargs)


    def save(self, directory=None, filename=None):

        if (filename is None) == (directory is None):
            raise Runtime("Either directory or filename must be given")

        if filename is None:
            uid = uuid4()
            filename = os.path.join(directory, "smrt-promise-%s.P" % uid)
            self.result_filename  = "smrt-result-%s.nc" % uid

        with open(filename, "wb") as f:
            pickle.dump(self, f)   

        return filename

# class RunPromiseBatch(object):

#     def __init__(self, promise):
#         self.promises = promises

#     def run(self):
#         return [promise.run() for promise in promises]



