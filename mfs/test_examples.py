from . import examples


def test_exmaple_permanent_magnets():
    examples.DEBUG = True
    examples.permanent_magnets()
    examples.plt.close("all")


def test_example_rectangular_coil_pair():
    examples.DEBUG = True
    examples.rectangular_coil_pair()
    examples.plt.close("all")


def test_example_circular_coil_pair():
    examples.DEBUG = True
    examples.circular_coil_pair()
    examples.plt.close("all")
