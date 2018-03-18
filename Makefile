venv:
	virtualenv venv --python=python3.6
	venv/bin/pip install -r requirements-minimal.txt
	touch venv/bin/activate
	export PYTHONPATH=`pwd`

data: venv
	mkdir taming_the_monster/data
	touch taming_the_monster/data/generated_data
	venv/bin/python taming_the_monster/generate_data/simulation.py

clean:
	rm -rf venv
	rm -rf taming_the_monster/data
