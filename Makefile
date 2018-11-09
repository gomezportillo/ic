PYTHON3 := $(shell command -v python3 2> /dev/null)
PIP3		:= $(pip3)
all: run

run:
ifndef PYTHON3
    $(error "Python3 is not installed. Please do so with apt-get install python3")
endif
	$(PYTHON3) src/mnist.py

install:
	apt-get install python3 python3-pip python3-tk graphviz
	$(PIP3) install -r requirements.txt

clean:
	find . -name '*__pycache__' -exec rm -vrf {} \;
