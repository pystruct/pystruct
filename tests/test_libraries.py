from pystruct.inference import get_installed

def test_pyqpbo() :
    import pyqpbo
    assert 'qpbo' in get_installed()

def test_ad3() :
    import ad3
    assert 'ad3' in get_installed()
