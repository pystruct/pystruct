from pystruct.inference import get_installed


def test_pyqpbo():
    import pyqpbo
    pyqpbo
    assert 'qpbo' in get_installed(['qpbo'])


def test_ad3():
    import ad3
    ad3
    assert 'ad3' in get_installed(['ad3'])

def test_ad3plus():
    import ad3
    ad3
    assert 'ad3+' in get_installed(['ad3+'])
    
    
