experiments : clean link

# Link to python package
link:
	ln -s ../IB IB

data:
	./helpers/get_data.sh

# Run experiment
quantize:
	python helpers/quantize.py SYN-Tanh 8
	python helpers/quantize.py SYN-ReLU 8
	python helpers/quantize.py MNIST-Bottleneck-2 8 0 20
binning:
	python helpers/binning.py SYN-Tanh
	python helpers/binning.py SYN-ReLU
bit-width:
	python helpers/quantize.py SYN-Tanh 4
	python helpers/quantize.py SYN-ReLU 4
	python helpers/quantize.py SYN-Tanh 32
	python helpers/quantize.py SYN-ReLU 32
quantize-prefit:
	python helpers/quantize.py SYN-Tanh 8 1000 20
	python helpers/quantize.py SYN-ReLU 8 1000 20
quantize-archs:
	python helpers/quantize.py MNIST-Bottleneck-4 8 0 20
	python helpers/quantize.py MNIST-HourGlass 8 0 20
	python helpers/quantize.py MNIST-10 8 0 10
	python helpers/quantize.py MNIST-Conv 8 0 10

# Create latex plots
plot-quantize:
	mkdir -p plots
	python helpers/plot.py quantize
	# I-planes
	cd helpers/latex; lualatex SYN_8b_ReLU_IP
	cd helpers/latex; lualatex SYN_8b_Tanh_IP
	cd helpers/latex; lualatex REAL_8b_BN_IP
	# Clean and move
	cd helpers/latex; rm -f SYN_8b_*.{log,aux}; rm -f REAL_8b_BN_IP.{log,aux}; rm -rf quantize
	mv helpers/latex/SYN_8b_*.pdf plots/
	mv helpers/latex/REAL_8b_BN_IP.pdf plots/
plot-binning:
	mkdir -p plots
	python helpers/plot.py binning
	cd helpers/latex; lualatex SYN_bin_ReLU_IP
	cd helpers/latex; lualatex SYN_bin_Tanh_IP
	cd helpers/latex; rm -f SYN_bin_*.{log,aux}; rm -rf binning
	mv helpers/latex/SYN_bin_*.pdf plots/
plot-bit-width:
	mkdir -p plots
	python helpers/plot.py bit-width
	cd helpers/latex; lualatex SYN_4b_IP
	cd helpers/latex; lualatex SYN_32b_IP
	cd helpers/latex; rm -f SYN_*.{log,aux}; rm -f REAL_8b_10_IP.{log,aux}; rm -rf bit-width
	mv helpers/latex/SYN_{4,32}b_*.pdf plots/
plot-prefit:
	mkdir -p plots
	python helpers/plot.py prefit
	cd helpers/latex; lualatex SYN_8b_prefit_IP
	cd helpers/latex; rm -f SYN_8b_prefit_IP.{log,aux}; rm -rf prefit
	mv helpers/latex/SYN_8b_prefit_IP.pdf plots/
plot-archs:
	mkdir -p plots
	python helpers/plot.py archs
	cd helpers/latex; lualatex REAL_8b_archs_IP 
	cd helpers/latex; rm -f REAL_8b_archs_IP.{log,aux}; rm -rf archs
	mv helpers/latex/REAL_8b_archs_IP.pdf plots/
plot-accuracy:
	mkdir -p plots
	python helpers/plot.py accuracy
	cd helpers/latex; lualatex SYN_Tanh_accuracy
	cd helpers/latex; lualatex SYN_ReLU_accuracy
	cd helpers/latex; lualatex REAL_accuracy
	cd helpers/latex; rm -f SYN_*.{log,aux}; rm -f REAL_accuracy.{log,aux} rm -rf accuracy
	mv helpers/latex/SYN_*.pdf plots/
	mv helpers/latex/REAL_accuracy.pdf plots/

# Clean
clean:
	rm -f IB
	rm -rf out
	rm -rf plots
	rm -rf data
