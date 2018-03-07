venv:
	virtualenv venv --python=python3.6
	venv/bin/pip install -r requirements-minimal.txt
	touch venv/bin/activate

clean:
	rm -rf venv
