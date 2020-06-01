


from smrt.core.interface import make_interface
from smrt import SMRTError


def test_make_interface_noargs():

	make_interface("flat")


# @raises(SMRTError)
# def test_make_interface_require_args():

# 	make_interface("geometrical_optics")


# def test_make_interface_with_args():

# 	make_interface("geometrical_optics", mean_square_slope=1)



# def test_make_interface_with_multiple_args():

# 	mss = [1, 2, 3]
# 	interface_broadcasted = make_interface("geometrical_optics", mean_square_slope=mss)

# 	assert (len(interface_broadcasted) == len(mss))
