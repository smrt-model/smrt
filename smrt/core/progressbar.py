# -*- python -*-
# -*- coding: utf-8 -*-
#
#  This file is part of the easydev software
#
#  Copyright (c) 2011-2017
#
#  File author(s): Thomas Cokelaer <cokelaer@gmail.com>
#
#  Distributed under the GPLv3 License.
#  See accompanying file LICENSE.txt or copy at
#      http://www.gnu.org/licenses/gpl-3.0.html
#
#  Website: https://github.com/cokelaer/easydev
#  Documentation: http://easydev-python.readthedocs.io
#
##############################################################################
"""A progress bar copied and adapted from pyMC code (dec 2014)"""
from __future__ import print_function

import sys
import time
import uuid


try:
    from IPython.core.display import HTML, Javascript, display
except ImportError:
    pass


__all__ = ['progress_bar', 'TextProgressBar', 'Progress']


class ProgressBar(object):
    def __init__(self, iterations, interval=None):
        self.iterations = iterations
        # if no interval provided, set it to 1%
        if interval is None:
            if iterations <= 100:
                interval = 1  # everything % of the data
            else:
                interval = int(iterations/100)
        self.interval = interval
        self.start = time.time()
        self.last = 0

    def _percentage(self, i):
        if self.iterations != 0:
            return 100 * i / float(self.iterations)
        else:
            # could be 100 ?
            return 100

    def _get_elapsed(self):
        return time.time() - self.start
    elapsed = property(_get_elapsed)


class TextProgressBar(ProgressBar):
    """Use :class:`Progress`"""

    def __init__(self, iterations, printer, width=40, interval=None):
        self.fill_char = '-'
        self.width = width
        self.printer = printer
        ProgressBar.__init__(self, iterations, interval=interval)

    def animate(self, i, dummy=None):
        # dummy=None is for back-compatibility
        if dummy is not None:
            print("second argument in easydev.progress_bar.animate is deprecated. Update your code")
        # +1 if i starts at 0 and finishes at N-1
        if divmod(i, self.interval)[1] != 0 and i != self.iterations:
            pass
        else:
            self.printer(self.progbar(i))

    def progbar(self, i):
        # +1 if i starts at 0 and finishes at N-1
        bar = self.bar(self._percentage(i))
        return "[%s] %i of %i complete in %.1f sec" % (
            bar, (i), self.iterations, round(self.elapsed, 1))

    def bar(self, percent):
        all_full = self.width - 2
        num_hashes = int(percent / 100 * all_full)

        bar = self.fill_char * num_hashes + ' ' * (all_full - num_hashes)

        info = '%d%%' % percent
        loc = (len(bar) - len(info)) // 2
        return replace_at(bar, info, loc, loc + len(info))


def replace_at(str, new, start, stop):
    return str[:start] + new + str[stop:]


def consoleprint(s):
    if sys.platform.lower().startswith('win'):
        print(s, '\r', end='')
    else:
        print('\r', s, end='')
        sys.stdout.flush()


def ipythonprint(s):
    print('\r', s, end='')
    sys.stdout.flush()


class IPythonNotebookPB(ProgressBar):
    """Use :class:`Progress`"""

    def __init__(self, iterations, interval=None):
        self.divid = str(uuid.uuid4())
        self.sec_id = str(uuid.uuid4())

        pb = HTML(
            """
            <div style="float: left; border: 1px solid black; width:500px">
              <div id="%s" style="background-color:blue; width:0%%">&nbsp;</div>
            </div>
            <label id="%s" style="padding-left: 10px;" text = ""/>
            """ % (self.divid, self.sec_id))
        display(pb)

        ProgressBar.__init__(self, iterations, interval=interval)

    def animate(self, i, dummy=None):
        if dummy is not None:
            print("second argument in easydev.progress_bar.animate is deprecated. Update your code")

        # +1 if i starts at 0 and finishes at N-1
        if divmod(i, self.interval)[1] != 0 and i != self.iterations:
            pass
        else:
            percentage = self._percentage(i)
            fraction = percentage
            display(
                Javascript("$('div#%s').width('%i%%')" %
                           (self.divid, percentage)))
            display(
                Javascript("$('label#%s').text('%i%% in %.1f sec')" %
                           (self.sec_id, fraction, round(self.elapsed, 1))))


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def progress_bar(iters, interval=None):
    """A progress bar for Python/IPython/IPython notebook

    :param int iters: number of iterations (steps in the loop)
    :param interval: number of intervals to use to update the progress bar (20
        by default)


    ::

        from easydev import progress_bar
        pb = progress_bar(10)
        for i in range(1,10):
            import time
            time.sleep(0.1)
            pb.animate(i)

    """
    if _run_from_ipython():
        if in_ipynb() is True:
            return IPythonNotebookPB(iters, interval=interval)
        else:
            return TextProgressBar(iters, printer=ipythonprint,
                                   interval=interval)
    else:
        return TextProgressBar(iters, printer=consoleprint,
                               interval=interval)


class Progress(object):
    """Generic progress bar for python, IPython, IPython notebook

    ::

        from easydev import Progress
        pb = Progress(100, interval=1)
        pb.animate(10)
    """

    def __init__(self, iters, interval=None):
        self.pb = progress_bar(iters, interval=interval)

    def animate(self, i):
        self.pb.animate(i)

    def _get_elapsed(self):
        return self.pb.elapsed
    elapsed = property(_get_elapsed)


def in_ipynb():
    """Checks if we are in an ipython notebook

    :return: True if in an ipython notebook otherwise returns False

    """
    try:
        cfg = get_ipython().config
        if 'parent_appname' in cfg['IPKernelApp'].keys() and \
                cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        elif "connection_file" in cfg['IPKernelApp'].keys():
            if "jupyter" in cfg['IPKernelApp']['connection_file']:
                return True
        return False
    except NameError:
        return False
