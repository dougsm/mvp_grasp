import time


class TimeIt:
    """
    Print nested timing information.
    """
    print_output = True
    last_parent = None
    level = -1

    def __init__(self, s):
        """
        Initialize the output.

        Args:
            self: (todo): write your description
            s: (int): write your description
        """
        self.s = s
        self.t0 = None
        self.t1 = None
        self.outputs = []
        self.parent = None

    def __enter__(self):
        """
        Enter the next time.

        Args:
            self: (todo): write your description
        """
        self.t0 = time.time()
        self.parent = TimeIt.last_parent
        TimeIt.last_parent = self
        TimeIt.level += 1

    def __exit__(self, t, value, traceback):
        """
        Prints the traceback.

        Args:
            self: (todo): write your description
            t: (todo): write your description
            value: (todo): write your description
            traceback: (todo): write your description
        """
        self.t1 = time.time()
        st = '%s%s: %0.1fms' % ('  ' * TimeIt.level, self.s, (self.t1 - self.t0)*1000)
        TimeIt.level -= 1

        if self.parent:
            self.parent.outputs.append(st)
            self.parent.outputs += self.outputs
        else:
            if TimeIt.print_output:
                print(st)
                for o in self.outputs:
                    print(o)
            self.outputs = []

        TimeIt.last_parent = self.parent
