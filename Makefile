all:
	python setup.py build_ext -i

coverage:
	nosetests --with-coverage --cover-html --cover-package=pystruct tests

test:
	nosetests -sv tests

clean:
	find | grep .pyc | xargs rm
