from pystruct.datasets import load_scene, load_letters, load_snakes


def test_dataset_loading():
    # test that we can read the datasets.
    load_scene()
    load_letters()
    load_snakes()
