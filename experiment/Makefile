experiments : clean link

# Link to python package
link:
	ln -s ../IB IB

data:
	./helpers/get_data.sh

# Run experiment
binning:
	python helpers/binning.py tanh
	python helpers/binning.py relu
quantize:
	python helpers/quantize.py tanh 8
	python helpers/quantize.py relu 8
quantize-extra:
	python helpers/quantize.py tanh 4
	python helpers/quantize.py relu 4
	python helpers/quantize.py tanh 32
	python helpers/quantize.py relu 32

# Create latex plots
plot-binning:
	mkdir -p plots
	python helpers/plot.py binning
	cd helpers/latex; lualatex binning_relu
	cd helpers/latex; lualatex binning_tanh
	cd helpers/latex; rm -f binning_*.{log,aux}; rm -rf binning
	mv helpers/latex/binning_*.pdf plots/
plot-quantize:
	mkdir -p plots
	python helpers/plot.py quantize
	cd helpers/latex; lualatex 8bit_relu
	cd helpers/latex; lualatex 8bit_tanh
	cd helpers/latex; lualatex 8bit_MNIST
	cd helpers/latex; rm -f 8bit_*.{log,aux}; rm -rf quantize
	mv helpers/latex/8bit_*.pdf plots/
plot-accuracy:
	mkdir -p plots
	python helpers/plot.py accuracy
	cd helpers/latex; lualatex accuracy 
	cd helpers/latex; rm -f accuracy.{log,aux}; rm -rf accuracy
	mv helpers/latex/accuracy.pdf plots/
plot-quantize-extra:
	mkdir -p plots
	python helpers/plot.py quantize-extra
	cd helpers/latex; lualatex 4bit
	cd helpers/latex; lualatex 4bit_accuracy
	cd helpers/latex; lualatex 32bit
	cd helpers/latex; lualatex 32bit_accuracy
	cd helpers/latex; rm -f 4bit*.{log,aux}; rm -f 32bit*.{log,aux}; rm -rf quantize-extra
	mv helpers/latex/{4,32}bit*.pdf plots/

# Clean
clean:
	rm -f IB
	rm -rf out
	rm -rf plots
