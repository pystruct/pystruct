from pystruct.inference import get_installed

def test_pyqpbo() :
    import pyqpbo
    assert 'qpbo' in get_installed()
