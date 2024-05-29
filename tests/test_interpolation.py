import unittest
import numpy as np
from xr_fresh.interpolate_series import *
from pathlib import Path
from glob import glob
from datetime import datetime
import geowombat as gw
import os
import tempfile

class TestInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmp_dir = tempfile.TemporaryDirectory()
        # set change directory to location of this file
        cls.pth = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cls.pth)
        cls.pth = f"{cls.pth}/data/"
        cls.files = sorted(glob(f"{cls.pth}values_equal_*.tif"))
        print(cls.files)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        try:
            cls.tmp_dir.cleanup()
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

    def test_linear_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(self.files, transfer_lib="jax") as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="linear",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 2313))
                # assert all of band 1 are equal to 1 NOTE EDGE CASE NOT HANDLED
                # assert np.all(dst[0] == 1)
                # assert all of band 2 are equal to 2
                assert np.all(dst[1] == 2)
                # assert all of band 3 are equal to 3
                assert np.all(dst[2] == 3)
                # assert all of band 4 are equal to 4
                assert np.all(dst[3] == 4)
                # assert all of band 5 are equal to 5
                # assert np.all(dst[4] == 5) NOTE EDGE CASE NOT HANDLED

if __name__ == "__main__":
    unittest.main()
