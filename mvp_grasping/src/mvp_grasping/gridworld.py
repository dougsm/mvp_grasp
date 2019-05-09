import numpy as np


class GridWorld:
    def __init__(self, bounds, resolution):
        if np.any(bounds[0, :] > bounds[1, :]):
            raise ValueError('Please make sure bounds[0, :] < bounds[1, :]')
        self.bounds = bounds
        self.res = resolution
        self.shape = np.ceil((bounds[1, :] - bounds[0, :] - 1e-6)/resolution).astype(np.int)
        self.map_offset = bounds[0, :]/resolution * -1
        self.maps = {}

    def add_grid(self, name, init_value=0, extra_dims=(),):
        shape = list(self.shape) + list(extra_dims)
        self.maps[name] = np.full(shape, init_value)

    def pos_to_cell(self, pos):
        """
        Convert a set of x, y positions to cell ids.
        :param pos: Nx2 array of positions
        :param ignore_outliers: If True, pos's outside the bounds will be ignored. If False, ValueError will be raised.
        :return: Nx2 array of cell ids
        """
        cell_ids = np.floor((pos / self.res) + self.map_offset).astype(np.int)

        cell_ids[:, 0] = np.clip(cell_ids[:, 0], 0, self.shape[1]-1)
        cell_ids[:, 1] = np.clip(cell_ids[:, 1], 0, self.shape[0]-1)

        # check_bounds = (cell_ids >= 0) & (cell_ids < self.shape)
        # if not np.all(check_bounds):
        #         raise ValueError('Position is outside of bounds.')
        return np.flip(cell_ids, axis=1)

    def cell_to_pos(self, cell_ids):
        return (np.flip(cell_ids, axis=1) - self.map_offset) * self.res + self.res/2

    def __getattr__(self, item):
        return self.maps[item]


if __name__ == '__main__':
    h = GridWorld(np.array([
        [-0.25, -0.8],
        [0.25, -0.3]
    ]), 0.005)

    h.add_grid('test')

    print(h.shape)
    print(h.map_offset)

    cell_ids = h.pos_to_cell(np.array([
        [-0.25, -0.8],
        [-0.0, -0.5],
        [+0.25, -0.3],
    ]))

    print(cell_ids)

    print(h.cell_to_pos(cell_ids))
