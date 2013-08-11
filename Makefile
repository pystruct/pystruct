all:
	python setup.py build_ext -i

coverage:
	nosetests --with-coverage --cover-html --cover-package=pystruct pystruct

test:
	nosetests -sv pystruct

clean:
	find | grep .pyc | xargs rm
