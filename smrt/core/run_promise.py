
import os
import glob
import pickle
import random
from uuid import uuid4
from .error import SMRTError
from .filelock import FileLock, Timeout


def honour_all_promises(directory_or_filename, save_result_to=None, show_progress=True, force_compute=True):
    """Honour many promises and save the results

    :param directory_or_filename: can be a directory, a filename or a list of them
    :param save_result_to: directory where to save the results. If None, the results are not saved. The results are always returned as a list by this function.
    :param show_progress: print progress of the calculation.
    :param force_computate: If False and if a result or lock file is present, the computation is skipped. The order of promise processing is randomized
     to allow more efficient parallel computation using many calls of this function on the same directory. A lock file is used between the start of a computation and 
     writting the result in order to prevent from running several times the same computation. If the process is interupted (e.g. walltime on clusters), the lock file may persist and prevent any future computation. In this case,
     lock files must be manually deleted.
     IF False, the `save_result_to` argument must be set to a valid directory where the results. 
"""

    if isinstance(directory_or_filename, str):
        directory_or_filename = [directory_or_filename]

    filename_list = []
    for item in directory_or_filename:
        if os.path.isdir(item):
            filename_list += glob.glob(os.path.join(item, "smrt-promise-*.P"))
        elif os.path.isfile(item):
            filename_list.append(item)
        else:
            raise SMRTError("directory_or_filename argument must be an existing directory or a filename or a list of them.")

    if not force_compute:
        random.shuffle(filename_list)

    if save_result_to is not None and not os.path.isdir(save_result_to):
        raise SMRTError("save_result_to must be an existing directory (or None).")

    result_list = []
    for filename in filename_list:
        if show_progress:
            print(filename)
        result = honour_promise(filename, save_result_to=save_result_to, force_compute=force_compute)
        if result is not None:
            result_list.append(result)

    if show_progress:
        print("Executed %i promise(s). Done!" % len(result_list))
    return result_list


def honour_promise(filename, save_result_to=None, force_compute=True):
    """Honour a promise and optionally save the result.

    :param filename: file name of the promise
    :param save_result_to: directory where to save the result.
    :param force_compute: see `honour_all_promise`.
"""

    promise = load_promise(filename)

    # determine the filename of the results
    if save_result_to is not None:
        if os.path.isdir(save_result_to):
            if getattr(promise, "result_filename", None) is None:
                raise SMRTError(
                    "promise has no predefined output filename and save_result_to is a directory. Either rebuild the promise or provide a file for save_result_to.")
            outfilename = os.path.join(save_result_to, promise.result_filename)
        elif os.path.isfile(save_result_to):
            outfilename = save_result_to
        else:
            raise SMRTError("save_result_to argument must be a directory or a filename")

    if force_compute is False:
        if save_result_to is None:
            raise SMRTError("save_result_to must be set to an existing directory when force_compute is False.")

        if os.path.exists(outfilename):
            return  # the result exist, no need to do the computation
        lock = FileLock(outfilename + ".lock", timeout=0)
        try:
            with lock:
                if os.path.exists(outfilename):  # check the result file has not been written between the first check and the lock acquisition.
                    return  # done!
                result = promise.run()
                result.save(outfilename)
        except Timeout:
            return  # another process is doing the computation according to the existence of the lock file.

    else:
        result = promise.run()
        if save_result_to is not None:
            result.save(outfilename)

    return result


def load_promise(filename):

    with open(filename, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, RunPromise):
        raise SMRTError("The file does not contain a SMRT promise")

    return obj


class RunPromise(object):

    def __init__(self, model, sensor, snowpack, kwargs):

        super().__init__()

        self.model = model
        self.sensor = sensor
        self.snowpack = snowpack
        self.kwargs = kwargs  # options and other optional arguments
        self.result_filename = None

    def run(self):

        return self.model.run(self.sensor, self.snowpack, **self.kwargs)

    def save(self, directory=None, filename=None):

        if (filename is None) == (directory is None):
            raise RuntimeError"Either directory or filename must be given")

        if filename is None:
            uid = uuid4()
            filename = os.path.join(directory, "smrt-promise-%s.P" % uid)
            self.result_filename = "smrt-result-%s.nc" % uid
        else:
            basename = os.path.basename(filename)
            if basename.startswith("smrt-promise-"):
                basename = "smrt-result-" + basename[13:]
            self.result_filename = os.path.splitext(basename)[0] + '.nc'

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        return filename

# class RunPromiseBatch(object):

#     def __init__(self, promise):
#         self.promises = promises

#     def run(self):
#         return [promise.run() for promise in promises]
